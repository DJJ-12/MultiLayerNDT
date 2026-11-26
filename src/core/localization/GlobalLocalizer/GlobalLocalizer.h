#ifndef __GLOBAL_LOCALIZER_H__
#define __GLOBAL_LOCALIZER_H__ 

#include <cstdint>
#include <cmath>
#include <string>

#include <Eigen/Core>
#include "GlobalLocalizer/ndt_omp/estimate_covariance.hpp"
#include "MapLoader/MapLoader.hpp"

namespace robot_localizer
{

// 定位质量评估阈值配置
struct QualityThresholds {
    double excellent_threshold = 4.0;           // 优秀质量阈值
    double good_threshold = 2.5;                // 良好质量阈值
    double fair_threshold = 1.5;                // 一般质量阈值
    int excellent_iteration_threshold = 20;     // 优秀质量的迭代次数阈值
};

struct LocalizationQuality {
    double transform_probability = 0.0;           // 变换概率
    double nearest_voxel_likelihood = 0.0;        // 最近体素似然度
    int iteration_num = 0;                        // 迭代次数
    bool is_reliable = false;                     // 是否可靠
    std::string quality_level = "unknown";       // 质量等级：excellent/good/fair/poor
    Eigen::Matrix<double, 6, 6> pose_covariance = Eigen::Matrix<double, 6, 6>::Identity();
    bool covariance_valid = false;

    // 评估定位质量（使用自定义阈值）
    void evaluate(const pclomp::NdtResult& ndt_result, const QualityThresholds& thresholds,
                  const pclomp::NdtCovarianceOptions& options, int max_iterations) {
        transform_probability = ndt_result.transform_probability;
        nearest_voxel_likelihood = ndt_result.nearest_voxel_transformation_likelihood;
        iteration_num = ndt_result.iteration_num;
        if (transform_probability > thresholds.excellent_threshold && 
            iteration_num < thresholds.excellent_iteration_threshold) {
            quality_level = "excellent";
            is_reliable = true;
        } else if (transform_probability > thresholds.good_threshold) {
            quality_level = "good";
            is_reliable = true;
        } else if (transform_probability > thresholds.fair_threshold) {
            quality_level = "fair";
            is_reliable = true;
        } else {
            quality_level = "poor";
            is_reliable = false;
        }
        const auto covariance_result = pclomp::estimate_pose_covariance_by_laplace_approximation(ndt_result, options);
        pose_covariance = covariance_result.pose_covariance;
        covariance_valid = covariance_result.covariance_valid && is_reliable &&
            std::isfinite(transform_probability) && std::isfinite(nearest_voxel_likelihood) &&
            iteration_num < max_iterations;
    }
};

class Localizer
{
private:
    using PointSource = pcl::PointXYZ;
    using PointTarget = pcl::PointXYZ;
    using NormalDistributionsTransform = pclomp::MultiGridNormalDistributionsTransform<PointSource, PointTarget>;
    std::shared_ptr<NormalDistributionsTransform> ndt_ptr_;
    std::shared_ptr<NormalDistributionsTransform> ndt_rough_ptr_;
    
    
    MapManager::MapLoader map_loader_;
    QualityThresholds quality_thresholds_;  // 定位质量评估阈值
    pclomp::NdtCovarianceOptions covariance_options_;

    void ResetNdt();
    pclomp::NdtResult AlignPose(const Eigen::Matrix4d &initial_pose_with_cov);

    Eigen::Matrix4d last_last_pose_ = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d last_pose_ = Eigen::Matrix4d::Identity();
    std::uint64_t register_frame_count_ = 0;
    
public:
    Localizer();

    // 设置定位质量评估阈值
    void SetQualityThresholds(const QualityThresholds& thresholds) { quality_thresholds_ = thresholds; }
    void SetCovarianceOptions(const pclomp::NdtCovarianceOptions& options) { covariance_options_ = options; }

    bool GetInitPose(const Eigen::Matrix4d& init_guess, Eigen::Matrix4d& align_pose, const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud, LocalizationQuality& quality);

    void ResetLocalizationState();

    bool RegisterFrame(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud,
        Eigen::Matrix4d& align_pose,
        LocalizationQuality& quality,
        std::uint64_t diagnostic_sequence = 0,
        double diagnostic_timestamp = 0.0,
        const Eigen::Matrix4d* external_initial_guess = nullptr);

    void SetStaticMap(std::string pcd_metadata_path, std::string pcd_directory)
    {
        map_loader_.Initialize(pcd_metadata_path, pcd_directory);
    }

    void MapReset();
};
}

#endif
