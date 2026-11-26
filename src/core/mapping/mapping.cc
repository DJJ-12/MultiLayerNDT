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
Mapping::Mapping(MappingOptions options) : options_(std::move(options)), ndt_(options_.ndt), backend_(options_.loop) {}

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
    std::map<int, CloudPtr> clouds;
    for (auto id : active_) {
        auto found = active_clouds_.find(static_cast<int>(id));
        CloudPtr world;
        if (found != active_clouds_.end()) world = found->second;
        else {
            world.reset(new PointCloudType);
            const auto& frame = data_.keyframes[id];
            const Eigen::Matrix4f transform = frame.pose.matrix().cast<float>();
            pcl::transformPointCloud(*frame.cloud, *world, transform);
        }
        clouds.emplace(static_cast<int>(id), world);
        *submap_ += *world;
    }
    active_clouds_ = std::move(clouds);
    ndt_.SetTarget(submap_);
    return true;
}

bool Mapping::UpdateGraph() {
    const bool closed = backend_.Update(data_.keyframes);
    if (!data_.keyframes.empty()) {
        const auto& last = data_.keyframes.back();
        map_from_odom_ = last.pose * last.odom_pose.inverse();
    }
    if (closed) { active_.clear(); active_clouds_.clear(); }
    return closed;
}

MappingResult Mapping::Process(const Scan& scan) {
    MappingResult result;
    if (!scan.cloud || scan.cloud->empty()) return result;
    if (!data_.trajectory.empty() && scan.stamp <= data_.trajectory.back().timestamp_) return result;
    result.graph_updated = UpdateGraph();
    SE3 odom_prediction = data_.trajectory.empty() ? SE3() : data_.trajectory.back().pose;
    Vec6d body_velocity = Vec6d::Zero();
    if (data_.trajectory.size() >= 2) {
        const auto& last = data_.trajectory.back();
        const auto& previous = data_.trajectory[data_.trajectory.size() - 2];
        body_velocity = (previous.pose.inverse() * last.pose).log() / (last.timestamp_ - previous.timestamp_);
        odom_prediction = last.pose * SE3::exp(body_velocity * (scan.stamp - last.timestamp_));
    }
    // 运动只在连续odom系计算；回环通过map<-odom作用于匹配初值，不产生虚假速度。
    result.prediction = map_from_odom_ * odom_prediction;
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
    NavState odometry = result.state;
    odometry.pose = map_from_odom_.inverse() * result.state.pose;
    if (!data_.trajectory.empty()) {
        const auto& last = data_.trajectory.back();
        odometry.velocity = (odometry.pose.translation() - last.pose.translation()) / (scan.stamp - last.timestamp_);
    }
    data_.trajectory.push_back(odometry);  // 弱帧也传播，不把回环位姿跳变混入运动历史。
    bool add = data_.keyframes.empty();
    if (!add && result.converged) {
        const auto& last = data_.keyframes.back();
        const SE3 change = last.pose.inverse() * result.state.pose;
        add = change.translation().norm() >= options_.keyframe_distance ||
            change.so3().log().norm() >= options_.keyframe_angle || scan.stamp - last.stamp >= options_.keyframe_time;
    }
    if (add) {
        const auto sampled = Downsample(result.scan, options_.keyframe_voxel);
        data_.keyframes.push_back({scan.stamp, result.state.pose, sampled, odometry.pose});
        result.graph_updated = UpdateGraph() || result.graph_updated;
        backend_.CheckLoop(data_.keyframes);
        result.keyframe_added = true;
    }
    result.state.pose = map_from_odom_ * odometry.pose;
    result.state.velocity = map_from_odom_.so3() * odometry.velocity;
    if (data_.keyframes.size() == 1 || result.graph_updated)
        result.map_updated = UpdateSubmap(result.state.pose) || result.map_updated;
    result.keyframe_count = data_.keyframes.size();
    result.loop_count = backend_.LoopCount();
    if (result.map_updated) result.local_map = active_clouds_;
    last_scan_ = result.scan;
    return result;
}

MappingResult Mapping::Finish() {
    MappingResult result;
    if (data_.trajectory.empty()) return result;
    backend_.CheckLoop(data_.keyframes, true);
    backend_.WaitForLoop();
    result.graph_updated = UpdateGraph();
    result.valid = true;
    result.state = data_.trajectory.back();
    result.state.pose = map_from_odom_ * result.state.pose;
    result.state.velocity = map_from_odom_.so3() * result.state.velocity;
    result.scan = last_scan_;
    result.keyframe_count = data_.keyframes.size();
    result.loop_count = backend_.LoopCount();
    result.map_updated = UpdateSubmap(result.state.pose);
    if (result.map_updated) result.local_map = active_clouds_;
    return result;
}

std::vector<NavState> Mapping::CorrectedTrajectory(const MapData& data) {
    auto trajectory = data.trajectory;
    if (data.keyframes.empty()) return trajectory;
    std::size_t id = 0;
    for (auto& state : trajectory) {
        while (id + 1 < data.keyframes.size() && data.keyframes[id + 1].stamp <= state.timestamp_) ++id;
        const auto& frame = data.keyframes[id];
        SE3 correction = frame.pose * frame.odom_pose.inverse();
        if (id + 1 < data.keyframes.size()) {
            const auto& next = data.keyframes[id + 1];
            const double ratio = std::clamp((state.timestamp_ - frame.stamp) / (next.stamp - frame.stamp), 0.0, 1.0);
            const SE3 next_correction = next.pose * next.odom_pose.inverse();
            correction = correction * SE3::exp(ratio * (correction.inverse() * next_correction).log());
        }
        state.pose = correction * state.pose;
        state.velocity = correction.so3() * state.velocity;
    }
    return trajectory;
}

bool Mapping::SaveMap(const MapData& data, const std::string& directory, double voxel_size) {
    if (data.keyframes.empty()) return false;
    std::filesystem::create_directories(directory);
    CloudPtr map(new PointCloudType), sampled(new PointCloudType);
    for (const auto& keyframe : data.keyframes) {
        PointCloudType world;
        const Eigen::Matrix4f transform = keyframe.pose.matrix().cast<float>();
        pcl::transformPointCloud(*keyframe.cloud, world, transform);
        *map += world;
    }
    pcl::VoxelGrid<PointType> filter;
    filter.setLeafSize(voxel_size, voxel_size, voxel_size);
    filter.setInputCloud(map); filter.filter(*sampled);
    if (pcl::io::savePCDFileBinary((std::filesystem::path(directory) / "map.pcd").string(), *sampled) != 0) return false;
    std::ofstream trajectory(std::filesystem::path(directory) / "trajectory.txt");
    trajectory << std::fixed << std::setprecision(9);
    for (const auto& state : CorrectedTrajectory(data)) {
        const auto& t = state.pose.translation();
        const auto q = state.pose.unit_quaternion();
        trajectory << state.timestamp_ << ' ' << t.x() << ' ' << t.y() << ' ' << t.z() << ' '
                   << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
    }
    return trajectory.good();
}
}  // namespace lightning
