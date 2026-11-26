#include "core/mapping/mapping.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

namespace lightning {
Mapping::Mapping(MappingOptions options) : options_(std::move(options)), ndt_(options_.ndt) {}

CloudPtr Mapping::Downsample(const CloudPtr& cloud, double size) const {
    CloudPtr sampled(new PointCloudType);
    pcl::VoxelGrid<PointType> filter;
    filter.setLeafSize(size, size, size);
    filter.setInputCloud(cloud);
    filter.filter(*sampled);
    return sampled;
}

bool Mapping::UpdateSubmap(const SE3& prediction) {
    std::vector<std::pair<double, std::size_t>> nearby;
    for (std::size_t i = 0; i < data_.keyframes.size(); ++i) {
        const double distance = (data_.keyframes[i].pose.translation() - prediction.translation()).squaredNorm();
        if (distance <= options_.submap_radius * options_.submap_radius) nearby.emplace_back(distance, i);
    }
    if (nearby.empty()) nearby.emplace_back(0.0, data_.keyframes.size() - 1);
    std::sort(nearby.begin(), nearby.end(), [](const auto& a, const auto& b) {
        return a.first == b.first ? a.second > b.second : a.first < b.first;
    });
    if (nearby.size() > static_cast<std::size_t>(options_.submap_keyframes)) nearby.resize(options_.submap_keyframes);
    std::vector<std::size_t> selected;
    for (const auto& item : nearby) selected.push_back(item.second);
    std::sort(selected.begin(), selected.end());
    if (selected == active_) return false;
    active_ = std::move(selected);
    submap_.reset(new PointCloudType);
    for (auto id : active_) *submap_ += *data_.keyframes[id].cloud;
    ndt_.SetTarget(submap_);
    return true;
}

MappingResult Mapping::Process(const Scan& scan) {
    MappingResult result;
    if (!scan.cloud || scan.cloud->empty()) return result;
    if (!data_.trajectory.empty() && scan.stamp <= data_.trajectory.back().timestamp_) return result;
    result.prediction = data_.trajectory.empty() ? SE3() : data_.trajectory.back().pose;
    Vec6d body_velocity = Vec6d::Zero();
    if (data_.trajectory.size() >= 2) {
        const auto& last = data_.trajectory.back();
        const auto& previous = data_.trajectory[data_.trajectory.size() - 2];
        body_velocity = (previous.pose.inverse() * last.pose).log() / (last.timestamp_ - previous.timestamp_);
        result.prediction = last.pose * SE3::exp(body_velocity * (scan.stamp - last.timestamp_));
    }
    result.scan.reset(new PointCloudType(*scan.cloud));
    // 纯激光恒速去畸变：统一到header.stamp，点内时间已由接收层转换为秒。
    if (scan.relative_times.size() == result.scan->size()) {
        for (std::size_t i = 0; i < result.scan->size(); ++i) {
            auto& point = result.scan->points[i];
            const Vec3d corrected = SE3::exp(body_velocity * scan.relative_times[i]) * point.getVector3fMap().cast<double>();
            point.x = corrected.x(); point.y = corrected.y(); point.z = corrected.z();
        }
    }
    const auto source = Downsample(result.scan, options_.ndt.source_voxel);
    if (source->empty()) return result;
    NdtFit fit;
    if (data_.keyframes.empty()) {
        fit.pose = SE3(); fit.valid = fit.converged = true;
    } else {
        result.map_updated = UpdateSubmap(result.prediction);
        fit = ndt_.Align(source, result.prediction);
    }
    result.valid = true;
    result.converged = fit.valid && fit.converged;
    result.state.timestamp_ = scan.stamp;
    result.state.pose = fit.valid ? fit.pose : result.prediction;
    result.state.confidence_ = fit.score;
    result.iterations = fit.iterations;
    result.layer_scores = std::move(fit.layer_scores);
    if (!data_.trajectory.empty()) {
        const auto& last = data_.trajectory.back();
        result.state.velocity = (result.state.pose.translation() - last.pose.translation()) / (scan.stamp - last.timestamp_);
    }
    data_.trajectory.push_back(result.state);  // 弱帧也传播轨迹，不因未保存关键帧而原地停住。
    bool add = data_.keyframes.empty();
    if (!add && result.converged) {
        const auto& last = data_.keyframes.back();
        const SE3 change = last.pose.inverse() * result.state.pose;
        add = change.translation().norm() >= options_.keyframe_distance ||
            change.so3().log().norm() >= options_.keyframe_angle || scan.stamp - last.stamp >= options_.keyframe_time;
    }
    if (add) {
        const auto sampled = Downsample(result.scan, options_.keyframe_voxel);
        CloudPtr world(new PointCloudType);
        const Eigen::Matrix4f transform = result.state.pose.matrix().cast<float>();
        pcl::transformPointCloud(*sampled, *world, transform);
        data_.keyframes.push_back({scan.stamp, result.state.pose, world});
        result.keyframe_added = true;
        if (data_.keyframes.size() == 1) result.map_updated = UpdateSubmap(result.state.pose);
    }
    result.keyframe_count = data_.keyframes.size();
    if (result.map_updated) for (auto id : active_) result.local_map.emplace(static_cast<int>(id), data_.keyframes[id].cloud);
    return result;
}

bool Mapping::SaveMap(const MapData& data, const std::string& directory, double voxel_size) {
    if (data.keyframes.empty()) return false;
    std::filesystem::create_directories(directory);
    CloudPtr map(new PointCloudType), sampled(new PointCloudType);
    for (const auto& keyframe : data.keyframes) *map += *keyframe.cloud;
    pcl::VoxelGrid<PointType> filter;
    filter.setLeafSize(voxel_size, voxel_size, voxel_size);
    filter.setInputCloud(map); filter.filter(*sampled);
    if (pcl::io::savePCDFileBinary((std::filesystem::path(directory) / "map.pcd").string(), *sampled) != 0) return false;
    std::ofstream trajectory(std::filesystem::path(directory) / "trajectory.txt");
    trajectory << std::fixed << std::setprecision(9);
    for (const auto& state : data.trajectory) {
        const auto& t = state.pose.translation();
        const auto q = state.pose.unit_quaternion();
        trajectory << state.timestamp_ << ' ' << t.x() << ' ' << t.y() << ' ' << t.z() << ' '
                   << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
    }
    return trajectory.good();
}
}  // namespace lightning
