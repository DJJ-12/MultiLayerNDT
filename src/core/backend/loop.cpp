// 回环链路移植自lightning-lm/src/core/lio_sam/loop.cpp。
#include "core/backend/pose_graph.h"
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>
#include <glog/logging.h>
#include <cmath>

namespace lightning {
void PoseGraph::loopClosureThread() {
    for (;;) {
        std::vector<Keyframe> snapshot;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] { return !running_ || requested_; });
            if (!running_) break;
            snapshot.swap(loop_snapshot_); requested_ = false; busy_ = true;
        }
        try {
            LoopConstraint loop;
            if (performLoopClosure(snapshot, loop)) {
                QueueLoop(loop);
                loopIndexContainer[loop.from] = loop.to;
                LOG(INFO) << "[后端][回环] current=" << loop.from << " previous=" << loop.to << " fitness=" << loop.fitness;
            }
        } catch (const std::exception& error) { LOG(ERROR) << "[后端][回环] " << error.what(); }
        { std::lock_guard<std::mutex> lock(mutex_); busy_ = false; }
        condition_.notify_all();
    }
}

bool PoseGraph::detectLoopClosureDistance(const std::vector<Keyframe>& keyframes, std::size_t& previous) {
    const auto current = keyframes.size() - 1;
    if (loopIndexContainer.count(current)) return false;
    CloudPtr poses(new PointCloudType);
    for (const auto& frame : keyframes) {
        PointType point; const auto& position = frame.pose.translation();
        point.x = position.x(); point.y = position.y(); point.z = position.z(); poses->push_back(point);
    }
    pcl::KdTreeFLANN<PointType> tree;
    tree.setInputCloud(poses);
    std::vector<int> ids; std::vector<float> distances;
    tree.radiusSearch(poses->back(), options_.historyKeyframeSearchRadius, ids, distances, 0);
    for (int id : ids) if (keyframes[current].stamp - keyframes[id].stamp > options_.historyKeyframeSearchTimeDiff) {
        previous = static_cast<std::size_t>(id); return true;
    }
    return false;
}

CloudPtr PoseGraph::loopFindNearKeyframes(const std::vector<Keyframe>& keyframes, std::size_t key, int search_num) {
    CloudPtr combined(new PointCloudType), sampled(new PointCloudType);
    for (int offset = -search_num; offset <= search_num; ++offset) {
        const auto id = static_cast<std::int64_t>(key) + offset;
        if (id < 0 || id >= static_cast<std::int64_t>(keyframes.size())) continue;
        // 历史目标不能包含当前帧或时间上过近的帧，避免ICP对自身点云产生假回环。
        if (search_num > 0 && keyframes.back().stamp - keyframes[id].stamp <= options_.historyKeyframeSearchTimeDiff) continue;
        PointCloudType world;
        const Eigen::Matrix4f transform = keyframes[id].pose.matrix().cast<float>();
        pcl::transformPointCloud(*keyframes[id].cloud, world, transform);
        *combined += world;
    }
    if (!combined->empty()) {
        pcl::VoxelGrid<PointType> filter;
        filter.setLeafSize(options_.mappingSurfLeafSize, options_.mappingSurfLeafSize, options_.mappingSurfLeafSize);
        filter.setInputCloud(combined); filter.filter(*sampled);
    }
    return sampled;
}

bool PoseGraph::performLoopClosure(const std::vector<Keyframe>& keyframes, LoopConstraint& loop) {
    std::size_t previous = 0;
    if (!detectLoopClosureDistance(keyframes, previous)) return false;
    const auto current = keyframes.size() - 1;
    const auto source = loopFindNearKeyframes(keyframes, current, 0);
    const auto target = loopFindNearKeyframes(keyframes, previous, options_.historyKeyframeSearchNum);
    if (source->size() < 300 || target->size() < 1000) return false;
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(options_.historyKeyframeSearchRadius * 2);
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);
    icp.setInputSource(source); icp.setInputTarget(target);
    PointCloudType aligned;
    icp.align(aligned);
    const double fitness = icp.getFitnessScore();
    const auto correction = icp.getFinalTransformation();
    if (!icp.hasConverged() || !std::isfinite(fitness) || fitness > options_.historyKeyframeFitnessScore || !correction.allFinite()) return false;
    // ICP返回float矩阵，先归一化四元数，避免浮点正交误差破坏Sophus的旋转约束。
    const Eigen::Quaterniond rotation(correction.block<3, 3>(0, 0).cast<double>());
    const SE3 corrected = SE3(rotation.normalized(), correction.block<3, 1>(0, 3).cast<double>()) * keyframes[current].pose;
    loop = {current, previous, corrected.inverse() * keyframes[previous].pose, fitness};
    return true;
}
}  // namespace lightning
