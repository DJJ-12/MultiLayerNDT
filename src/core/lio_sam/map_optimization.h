#pragma once

#include "common/struct.h"
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/geometry/Pose3.h>
#include <condition_variable>
#include <thread>

namespace lightning {
// 从原mapOptimization提取的纯位姿图后端；不依赖LOAM前端、IMU或ROS节点。
// iSAM2只在调用线程更新，回环线程仅提交相对位姿约束。
class PoseGraph {
public:
    explicit PoseGraph(LoopOptions options);
    ~PoseGraph();
    bool Update(std::vector<Keyframe>& keyframes);  // 返回本次是否应用回环。
    void CheckLoop(const std::vector<Keyframe>& keyframes, bool force = false);
    void WaitForLoop();
    void QueueLoop(const LoopConstraint& loop);
    std::size_t LoopCount() const { return loop_count_; }

private:
    static gtsam::Pose3 ToGtsam(const SE3& pose);
    static SE3 FromGtsam(const gtsam::Pose3& pose);
    void loopClosureThread();
    bool performLoopClosure(const std::vector<Keyframe>& keyframes, LoopConstraint& loop);
    bool detectLoopClosureDistance(const std::vector<Keyframe>& keyframes, std::size_t& previous);
    CloudPtr loopFindNearKeyframes(const std::vector<Keyframe>& keyframes, std::size_t key, int search_num);

    LoopOptions options_;
    std::unique_ptr<gtsam::ISAM2> isam_;
    std::size_t graph_size_ = 0, loop_count_ = 0;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::thread loop_thread_;
    bool running_ = true, requested_ = false, busy_ = false;
    double last_check_stamp_ = -1.0;
    std::vector<Keyframe> loop_snapshot_;  // 只复制位姿和不可变点云指针，不复制历史点云。
    std::vector<LoopConstraint> loop_queue_;
    std::map<std::size_t, std::size_t> loopIndexContainer;  // 仅回环线程使用。
};
}  // namespace lightning
