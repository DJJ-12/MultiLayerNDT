#pragma once

#include <cstddef>
#include <mutex>
#include <functional>
#include <memory>
#include <string>
#include <cstdint>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "common/eigen_types.h"
#include "common/std_types.h"
#include "core/localization/GlobalLocalizer/GlobalLocalizer.h"
#include "common/struct.h"
#include "livox_ros_driver2/msg/custom_msg.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

namespace lightning {
namespace loc {

class Localization {
   public:
    struct Options {
        SE3 T_base_lidar_ = SE3();
    };

    explicit Localization(Options options);
    ~Localization() = default;

    bool Init(const std::string& yaml_path, const std::string& global_map_path);

    LocalizationFrameOutcome ProcessLidarMsg(
        const sensor_msgs::msg::PointCloud2::SharedPtr laser_msg,
        const LocalizationInputDiagnostic& diagnostic = {});
    LocalizationFrameOutcome ProcessLivoxLidarMsg(
        const livox_ros_driver2::msg::CustomMsg::SharedPtr laser_msg,
        const LocalizationInputDiagnostic& diagnostic = {});

    bool SetExternalPose(const Eigen::Quaterniond& q, const Eigen::Vector3d& t);
    void SetPredictionPose(const SE3& pose);
    void Finish();

    using ResultCallback = std::function<void(const LocalizationResult& result)>;
    using XYZCloud = pcl::PointCloud<pcl::PointXYZ>;

    void SetResultCallback(ResultCallback&& callback);
    void MarkPoor(const std::string& message);
    LocalizationResult GetLatestResult() const;

   private:
    bool TryInitializeWithCurrentCloud(const XYZCloud::Ptr& input_cloud, double timestamp);
    pcl::PointCloud<pcl::PointXYZ>::Ptr ConvertToBaseCloud(
        const sensor_msgs::msg::PointCloud2& msg,
        const Mat4f& T_base_lidar_matrix_f) const;
    pcl::PointCloud<pcl::PointXYZ>::Ptr ConvertToBaseCloud(
        const livox_ros_driver2::msg::CustomMsg& msg,
        const Mat4f& T_base_lidar_matrix_f) const;
    LocalizationFrameOutcome ProcessLocalizationCloud(const XYZCloud::Ptr& cloud, double timestamp, std::uint64_t diagnostic_sequence);
    void PublishResult(const Eigen::Matrix4d& pose, double timestamp,
                       const robot_localizer::LocalizationQuality& quality, const char* message = nullptr);
    static SE3 Matrix4dToSE3(const Eigen::Matrix4d& pose);

    std::mutex global_mutex_;
    std::mutex localizer_mutex_;
    Options options_;

    robot_localizer::Localizer localizer_;

    bool map_loaded_ = false;
    bool localization_inited_ = false;
    bool has_initial_pose_for_next_cloud_ = false;
    bool init_in_progress_ = false;

    Eigen::Matrix4d initial_pose_for_next_cloud_ = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d prediction_pose_ = Eigen::Matrix4d::Identity();
    bool has_prediction_pose_ = false;
    Mat4f T_base_lidar_matrix_f_ = Mat4f::Identity();

    robot_localizer::QualityThresholds quality_thresholds_;

    LocalizationResult loc_result_;
    mutable std::mutex loc_result_mutex_;

    ResultCallback result_callback_;

};

}  // namespace loc
}  // namespace lightning
