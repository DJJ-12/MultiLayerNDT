#include "core/backend/pose_graph.h"
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <glog/logging.h>
#include <algorithm>

namespace lightning {
PoseGraph::PoseGraph(LoopOptions options) : options_(options) {
    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.1;
    parameters.relinearizeSkip = 1;
    isam_ = std::make_unique<gtsam::ISAM2>(parameters);
    if (options_.loopClosureEnableFlag) loop_thread_ = std::thread(&PoseGraph::loopClosureThread, this);
}
PoseGraph::~PoseGraph() {
    { std::lock_guard<std::mutex> lock(mutex_); running_ = false; }
    condition_.notify_all();
    if (loop_thread_.joinable()) loop_thread_.join();
}
gtsam::Pose3 PoseGraph::ToGtsam(const SE3& pose) {
    return gtsam::Pose3(gtsam::Rot3(pose.rotationMatrix()), pose.translation());
}
SE3 PoseGraph::FromGtsam(const gtsam::Pose3& pose) { return SE3(pose.rotation().matrix(), pose.translation()); }

bool PoseGraph::Update(std::vector<Keyframe>& keyframes) {
    std::vector<LoopConstraint> loops;
    { std::lock_guard<std::mutex> lock(mutex_); loops.swap(loop_queue_); }
    if (graph_size_ == keyframes.size() && loops.empty()) return false;
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;
    for (std::size_t id = graph_size_; id < keyframes.size(); ++id) {
        const auto& frame = keyframes[id];
        if (id == 0) {
            // 纯激光没有GPS/IMU绝对先验，固定首帧六自由度，去掉图的坐标系自由度。
            auto priorNoise = gtsam::noiseModel::Diagonal::Variances(Vec6d::Constant(1e-6));
            graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, ToGtsam(frame.pose), priorNoise));
        } else {
            // 沿用原addOdomFactor的方差，顺序为旋转rad²、平移m²。
            auto odometryNoise = gtsam::noiseModel::Diagonal::Variances(
                (Vec6d() << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            const SE3 relative = keyframes[id - 1].odom_pose.inverse() * frame.odom_pose;
            graph.add(gtsam::BetweenFactor<gtsam::Pose3>(id - 1, id, ToGtsam(relative), odometryNoise));
        }
        initial.insert(id, ToGtsam(frame.pose));
    }
    for (const auto& loop : loops) {
        // 沿用原addLoopFactor：ICP均方误差作为回环噪声，零误差时保留数值下限。
        auto noise = gtsam::noiseModel::Diagonal::Variances(Vec6d::Constant(std::max(loop.fitness, 1e-6)));
        graph.add(gtsam::BetweenFactor<gtsam::Pose3>(loop.from, loop.to, ToGtsam(loop.relative_pose), noise));
        LOG(INFO) << "[后端][回环入图] from=" << loop.from << " to=" << loop.to << " fitness=" << loop.fitness;
    }
    isam_->update(graph, initial);
    isam_->update();
    if (!loops.empty()) for (int i = 0; i < 5; ++i) isam_->update();
    const auto estimate = isam_->calculateEstimate();
    // 普通里程计只新增末端；回环后像原correctPoses一样更新全部关键帧。
    const std::size_t first = loops.empty() ? graph_size_ : 0;
    for (std::size_t id = first; id < keyframes.size(); ++id) keyframes[id].pose = FromGtsam(estimate.at<gtsam::Pose3>(id));
    graph_size_ = keyframes.size();
    loop_count_ += loops.size();
    if (!loops.empty()) LOG(INFO) << "[后端][图优化] keyframes=" << graph_size_ << " loops=" << loop_count_;
    return !loops.empty();
}

void PoseGraph::QueueLoop(const LoopConstraint& loop) {
    std::lock_guard<std::mutex> lock(mutex_);
    loop_queue_.push_back(loop);
}
void PoseGraph::CheckLoop(const std::vector<Keyframe>& keyframes, bool force) {
    if (!options_.loopClosureEnableFlag || keyframes.size() < 2) return;
    std::lock_guard<std::mutex> lock(mutex_);
    const double stamp = keyframes.back().stamp;
    if (!force && last_check_stamp_ >= 0 && stamp - last_check_stamp_ < 1.0 / options_.loopClosureFrequency) return;
    last_check_stamp_ = stamp;
    loop_snapshot_ = keyframes;
    requested_ = true;
    condition_.notify_all();
}
void PoseGraph::WaitForLoop() {
    if (!options_.loopClosureEnableFlag) return;
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return !requested_ && !busy_; });
}
}  // namespace lightning
