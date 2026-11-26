#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <cstdint>
#include <string>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "common/eigen_types.h"

namespace lightning {
using PointType = pcl::PointXYZI;
using PointCloudType = pcl::PointCloud<PointType>;
using CloudPtr = PointCloudType::Ptr;
using UL = std::unique_lock<std::mutex>;

// 接收层已经把点内时间整理为相对header.stamp的秒数。
struct Scan {
    double stamp = 0.0;
    CloudPtr cloud;
    std::vector<double> relative_times;
};

// UI只需要这些状态，不引入滤波器和IMU状态模型。
struct NavState {
    // confidence_沿用UI接口名称，保存联合NDT评分，不表示标定后的概率。
    double timestamp_ = 0.0, confidence_ = 0.0;
    SE3 pose;
    Vec3d velocity = Vec3d::Zero();
    SE3 GetPose() const { return pose; }
    Vec3d GetVel() const { return velocity; }
    Vec3d Getba() const { return Vec3d::Zero(); }
    Vec3d Getbg() const { return Vec3d::Zero(); }
};

struct NdtOptions {
    std::vector<double> resolutions{2.0, 1.0, 0.5};
    std::vector<double> weights{1.0, 1.0, 1.0};
    int max_iterations = 30, num_threads = 4;
    double epsilon = 0.005, max_step = 0.3, source_voxel = 0.3;
};

struct NdtFit {
    SE3 pose;
    bool valid = false, converged = false;
    int iterations = 0;
    double score = 0.0;
    std::vector<double> layer_scores;
};

// 沿用原LIO-SAM回环参数名；频率按雷达时间控制，离线快速读包也会检查回环。
struct LoopOptions {
    bool loopClosureEnableFlag = true;
    double loopClosureFrequency = 1.0;
    double historyKeyframeSearchRadius = 15.0, historyKeyframeSearchTimeDiff = 30.0;
    int historyKeyframeSearchNum = 25;
    double historyKeyframeFitnessScore = 0.3, mappingSurfLeafSize = 0.4;
};

struct LoopConstraint {
    std::size_t from = 0, to = 0;
    SE3 relative_pose;
    double fitness = 0.0;
};

struct MappingOptions {
    NdtOptions ndt;
    LoopOptions loop;
    double keyframe_distance = 1.0, keyframe_angle = 0.2, keyframe_time = 2.0;
    double submap_radius = 50.0, keyframe_voxel = 0.15;
    int submap_keyframes = 20;
};

struct Keyframe {
    double stamp = 0.0;
    SE3 pose;
    CloudPtr cloud;  // 自身雷达系点云只存一份，回环后按新位姿重建地图。
    SE3 odom_pose;   // 连续前端位姿；图优化不能改变里程计边或运动预测。
};

struct MappingResult {
    NavState state;
    SE3 prediction;
    CloudPtr scan;
    bool valid = false, converged = false, keyframe_added = false, map_updated = false;
    bool graph_updated = false;
    std::size_t keyframe_count = 0, loop_count = 0;
    int iterations = 0;
    std::vector<double> layer_scores;
    std::map<int, CloudPtr> local_map;
};

struct MapData {
    std::vector<Keyframe> keyframes;
    std::vector<NavState> trajectory;  // 连续odom系；显示和保存时应用关键帧图优化校正。
};
}  // namespace lightning
