#include "core/localization/localization.h"

#include <glog/logging.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/time.hpp>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <string>
#include <vector>

namespace lightning::loc {

Localization::Localization(Options options) : options_(options) {}

bool Localization::Init(const std::string& yaml_path, const std::string& global_map_path) {
    Finish();

    std::lock_guard<std::mutex> loc_lock(localizer_mutex_);
    UL lock(global_mutex_);

    YAML::Node yaml_node = YAML::LoadFile(yaml_path);

    // Base ← LiDAR 外参：[tx, ty, tz, roll, pitch, yaw]，RPY 顺序，弧度。
    const auto v = yaml_node["common"]["BaseLidarTransform"].as<std::vector<double>>();
    Vec3d t_base_lidar(v[0], v[1], v[2]);
    Mat3d R_base_lidar = (Eigen::AngleAxisd(v[5], Vec3d::UnitZ())
                       * Eigen::AngleAxisd(v[4], Vec3d::UnitY())
                       * Eigen::AngleAxisd(v[3], Vec3d::UnitX())).toRotationMatrix();
    Quatd q_base_lidar(R_base_lidar);
    q_base_lidar.normalize();
    options_.T_base_lidar_ = SE3(q_base_lidar, t_base_lidar);
    T_base_lidar_matrix_f_ = options_.T_base_lidar_.matrix().cast<float>();
    LOG(INFO) << "[BASE_LIDAR] T_base_lidar trans="
              << options_.T_base_lidar_.translation().transpose();

    const auto quality = yaml_node["localization"]["quality"];
    quality_thresholds_.excellent_threshold = quality["excellent_threshold"].as<double>();
    quality_thresholds_.good_threshold = quality["good_threshold"].as<double>();
    quality_thresholds_.fair_threshold = quality["fair_threshold"].as<double>();
    quality_thresholds_.excellent_iteration_threshold =
        quality["excellent_iteration_threshold"].as<int>();

    pclomp::NdtCovarianceOptions covariance_options;
    const auto covariance = yaml_node["localization"]["ndt_covariance"];
    // 旧配置没有ndt_covariance段时沿用结构体默认值，避免读取缺失键抛异常。
    auto covariance_value = [&covariance](const char* key, double fallback) {
        const auto node = covariance[key];
        return node ? node.as<double>() : fallback;
    };
    covariance_options.information_scale = covariance_value("information_scale", covariance_options.information_scale);
    covariance_options.min_information_eigenvalue = covariance_value("min_information_eigenvalue", covariance_options.min_information_eigenvalue);
    covariance_options.max_information_eigenvalue = covariance_value("max_information_eigenvalue", covariance_options.max_information_eigenvalue);
    covariance_options.min_translation_std = covariance_value("min_translation_std", covariance_options.min_translation_std);
    covariance_options.max_translation_std = covariance_value("max_translation_std", covariance_options.max_translation_std);
    covariance_options.min_rotation_std_rad = covariance_value("min_rotation_std_deg", covariance_options.min_rotation_std_rad * 180.0 / M_PI) * M_PI / 180.0;
    covariance_options.max_rotation_std_rad = covariance_value("max_rotation_std_deg", covariance_options.max_rotation_std_rad * 180.0 / M_PI) * M_PI / 180.0;

    const std::string metadata_path =
        global_map_path + "/BlockMap/pointcloud_map_metadata.yaml";
    const std::string pcd_directory = global_map_path + "/BlockMap/pointcloud_map";

    localizer_.SetStaticMap(metadata_path, pcd_directory);
    localizer_.SetQualityThresholds(quality_thresholds_);
    localizer_.SetCovarianceOptions(covariance_options);

    map_loaded_ = true;
    localization_inited_ = false;
    has_initial_pose_for_next_cloud_ = false;
    init_in_progress_ = false;
    initial_pose_for_next_cloud_ = Eigen::Matrix4d::Identity();
    prediction_pose_ = Eigen::Matrix4d::Identity();
    has_prediction_pose_ = false;
    {
        UL lock_result(loc_result_mutex_);
        loc_result_ = LocalizationResult();
    }
    return true;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Localization::ConvertToBaseCloud(
    const sensor_msgs::msg::PointCloud2& msg,
    const Mat4f& T_base_lidar_matrix_f) const {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(msg, *cloud);

    std::vector<int> nan_indices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, nan_indices);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_base(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(
        *cloud, *cloud_base, T_base_lidar_matrix_f);
    cloud_base->header = cloud->header;
    cloud_base->header.frame_id = "base_link";
    cloud_base->width = cloud_base->size();
    cloud_base->height = 1;
    cloud_base->is_dense = false;
    return cloud_base;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Localization::ConvertToBaseCloud(
    const livox_ros_driver2::msg::CustomMsg& msg,
    const Mat4f& T_base_lidar_matrix_f) const {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    cloud->header.stamp = rclcpp::Time(msg.header.stamp).nanoseconds();
    cloud->header.frame_id = msg.header.frame_id;
    cloud->reserve(msg.points.size());
    for (const auto& pt : msg.points) {
        pcl::PointXYZ xyz;
        xyz.x = pt.x;
        xyz.y = pt.y;
        xyz.z = pt.z;
        cloud->push_back(xyz);
    }

    std::vector<int> nan_indices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, nan_indices);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_base(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(
        *cloud, *cloud_base, T_base_lidar_matrix_f);
    cloud_base->header = cloud->header;
    cloud_base->header.frame_id = "base_link";
    cloud_base->width = cloud_base->size();
    cloud_base->height = 1;
    cloud_base->is_dense = false;
    return cloud_base;
}

LocalizationFrameOutcome Localization::ProcessLidarMsg(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg,
    const LocalizationInputDiagnostic& diagnostic) {
    if (diagnostic.pipeline_sequence == 1) LOG(INFO) << "[Localization] first PointCloud2 received, points=" << static_cast<size_t>(msg->width) * msg->height;
    Mat4f T_base_lidar_matrix_f;
    {
        UL lock(global_mutex_);
        if (!map_loaded_) {
            LOG(ERROR) << "[Localization] 地图未加载";
            return LocalizationFrameOutcome::MAP_NOT_LOADED;
        }
        T_base_lidar_matrix_f = T_base_lidar_matrix_f_;
    }

    auto cloud = ConvertToBaseCloud(*msg, T_base_lidar_matrix_f);
    return ProcessLocalizationCloud(
        cloud, rclcpp::Time(msg->header.stamp).seconds(), diagnostic.pipeline_sequence);
}

LocalizationFrameOutcome Localization::ProcessLivoxLidarMsg(
    const livox_ros_driver2::msg::CustomMsg::SharedPtr msg,
    const LocalizationInputDiagnostic& diagnostic) {
    if (diagnostic.pipeline_sequence == 1) LOG(INFO) << "[Localization] first Livox cloud received, points=" << msg->points.size();
    Mat4f T_base_lidar_matrix_f;
    {
        UL lock(global_mutex_);
        if (!map_loaded_) {
            LOG(ERROR) << "[Localization] 地图未加载";
            return LocalizationFrameOutcome::MAP_NOT_LOADED;
        }
        T_base_lidar_matrix_f = T_base_lidar_matrix_f_;
    }

    auto cloud = ConvertToBaseCloud(*msg, T_base_lidar_matrix_f);
    return ProcessLocalizationCloud( cloud, rclcpp::Time(msg->header.stamp).seconds(), diagnostic.pipeline_sequence);
}

LocalizationFrameOutcome Localization::ProcessLocalizationCloud( const XYZCloud::Ptr& cloud, double timestamp, std::uint64_t diagnostic_sequence) {
    if (!cloud || cloud->empty()) {
        LOG(ERROR) << "[Localization] 点云转换后为空";
        return LocalizationFrameOutcome::EMPTY_AFTER_CONVERT;
    }

    bool initialized = false;
    bool has_init_pose = false;
    bool initializing = false;
    {
        UL lock(global_mutex_);
        initialized = localization_inited_;
        has_init_pose = has_initial_pose_for_next_cloud_;
        initializing = init_in_progress_;
    }

    if (!initialized) {
        if (has_init_pose && !initializing && TryInitializeWithCurrentCloud(cloud, timestamp)) {
            return LocalizationFrameOutcome::INITIALIZED_WITH_FRAME;
        }
        return LocalizationFrameOutcome::WAITING_INITIAL_POSE;
    }
    if (initializing) {
        return LocalizationFrameOutcome::INITIALIZATION_IN_PROGRESS;
    }

    XYZCloud::Ptr current_cloud(new XYZCloud);
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setLeafSize(2.0f, 2.0f, 2.0f);
    voxel_grid.setInputCloud(cloud);
    voxel_grid.filter(*current_cloud);
    if (current_cloud->empty()) {
        LOG(ERROR) << "[Localization] 体素滤波后点云为空";
        return LocalizationFrameOutcome::EMPTY_AFTER_VOXEL;
    }

    std::lock_guard<std::mutex> loc_lock(localizer_mutex_);
    Eigen::Matrix4d external_initial_guess = Eigen::Matrix4d::Identity();
    const Eigen::Matrix4d* external_initial_guess_ptr = nullptr;
    {
        UL lock(global_mutex_);
        if (!localization_inited_ || init_in_progress_) {
            return LocalizationFrameOutcome::STATE_NOT_READY;
        }
        if (has_prediction_pose_) {
            external_initial_guess = prediction_pose_;
            external_initial_guess_ptr = &external_initial_guess;
            has_prediction_pose_ = false;
        }
    }

    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    XYZCloud::Ptr cloud_reg(new XYZCloud);
    robot_localizer::LocalizationQuality quality;
    localizer_.RegisterFrame(
        current_cloud, cloud_reg, pose, quality, diagnostic_sequence, timestamp, external_initial_guess_ptr);

    PublishResult(pose, timestamp, quality);
    return LocalizationFrameOutcome::NDT_EXECUTED;
}

bool Localization::TryInitializeWithCurrentCloud(const XYZCloud::Ptr& input_cloud, double timestamp) {
    LOG(INFO) << "[INIT] initialization entered, points=" << input_cloud->size();
    std::lock_guard<std::mutex> loc_lock(localizer_mutex_);

    Eigen::Matrix4d init_guess = Eigen::Matrix4d::Identity();
    {
        UL lock(global_mutex_);
        if (!map_loaded_ ||
            !has_initial_pose_for_next_cloud_ ||
            localization_inited_ ||
            init_in_progress_) {
            return false;
        }
        init_in_progress_ = true;
        init_guess = initial_pose_for_next_cloud_;
    }

    Eigen::Matrix4d aligned_pose = Eigen::Matrix4d::Identity();
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(
        new pcl::PointCloud<pcl::PointXYZ>());
    robot_localizer::LocalizationQuality quality;

    localizer_.GetInitPose(
        init_guess, aligned_pose, input_cloud, output_cloud, quality);

    {
        UL lock(global_mutex_);
        localization_inited_ = true;
        has_initial_pose_for_next_cloud_ = false;
        init_in_progress_ = false;
    }

    const std::string message = "localization initialized: " + quality.quality_level;
    PublishResult(aligned_pose, timestamp, quality, message.c_str());
    return true;
}

void Localization::Finish() {
    LOG(INFO) << "[Localization::Finish] begin";
    {
        std::lock_guard<std::mutex> loc_lock(localizer_mutex_);
        localizer_.MapReset();
    }

    {
        UL lock(global_mutex_);
        map_loaded_ = false;
        localization_inited_ = false;
        has_initial_pose_for_next_cloud_ = false;
        init_in_progress_ = false;
    }
    LOG(INFO) << "[Localization::Finish] 完成";
}

bool Localization::SetExternalPose(const Eigen::Quaterniond& q, const Eigen::Vector3d& t) {
    Eigen::Matrix4d init_guess = Eigen::Matrix4d::Identity();
    init_guess.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
    init_guess.block<3, 1>(0, 3) = t;

    {
        std::lock_guard<std::mutex> loc_lock(localizer_mutex_);
        {
            UL lock(global_mutex_);
            initial_pose_for_next_cloud_ = init_guess;
            has_initial_pose_for_next_cloud_ = true;
            localization_inited_ = false;
            init_in_progress_ = false;
            prediction_pose_ = Eigen::Matrix4d::Identity();
            has_prediction_pose_ = false;
        }
        localizer_.ResetLocalizationState();
    }

    {
        UL lock_result(loc_result_mutex_);
        loc_result_ = LocalizationResult();
        loc_result_.valid_ = true;
        loc_result_.status_ = LocalizationStatus::INITIALIZING;
        loc_result_.pose_ = Matrix4dToSE3(init_guess);
        loc_result_.message_ = "initial pose accepted, waiting for next cloud";
    }

    LOG(INFO) << "[SET_LOCATION] accepted new initial pose x=" << init_guess(0, 3)
              << ", y=" << init_guess(1, 3)
              << ", z=" << init_guess(2, 3);
    return true;
}

void Localization::SetPredictionPose(const SE3& pose) {
    UL lock(global_mutex_);
    prediction_pose_ = pose.matrix();
    has_prediction_pose_ = true;
}


void Localization::PublishResult(const Eigen::Matrix4d& pose, double timestamp,
                                 const robot_localizer::LocalizationQuality& quality, const char* message) {
    LocalizationResult result;
    result.timestamp_ = timestamp;
    result.valid_ = true;
    result.localization_valid_ = quality.is_reliable;
    result.pose_ = Matrix4dToSE3(pose);
    result.confidence_ = quality.transform_probability;
    result.status_ = quality.is_reliable ? LocalizationStatus::GOOD : LocalizationStatus::FAIL;
    result.reliable_ = quality.is_reliable;
    result.tp_ = quality.transform_probability;
    result.nvtl_ = quality.nearest_voxel_likelihood;
    result.iterations_ = quality.iteration_num;
    result.pose_covariance_ = quality.pose_covariance;
    result.covariance_valid_ = quality.covariance_valid;
    result.message_ = message ? message : (quality.is_reliable ? "localization reliable" : "localization not reliable");
    {
        UL lock_result(loc_result_mutex_);
        loc_result_ = result;
    }

    if (result_callback_) {
        result_callback_(result);
    }
}

SE3 Localization::Matrix4dToSE3(const Eigen::Matrix4d& pose) {
    Mat3d rotation = pose.block<3, 3>(0, 0);
    Quatd q(rotation);
    q.normalize();
    return SE3(q, pose.block<3, 1>(0, 3));
}

void Localization::SetResultCallback(Localization::ResultCallback&& callback) {
    result_callback_ = std::move(callback);
}

void Localization::MarkPoor(const std::string& message) {
    UL lock_result(loc_result_mutex_);
    auto& result = loc_result_;
    result.valid_ = true;
    result.localization_valid_ = false;
    result.status_ = LocalizationStatus::FAIL;
    result.confidence_ = 0.0;
    result.reliable_ = false;
    result.tp_ = 0.0;
    result.nvtl_ = 0.0;
    result.iterations_ = 0;
    result.message_ = message.empty() ? "localization poor" : message;
}

LocalizationResult Localization::GetLatestResult() const {
    UL lock_result(loc_result_mutex_);
    return loc_result_;
}

}  // namespace lightning::loc
