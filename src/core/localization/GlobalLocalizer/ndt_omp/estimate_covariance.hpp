// Copyright 2023 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef AUTOWARE__NDT_SCAN_MATCHER__NDT_OMP__ESTIMATE_COVARIANCE_HPP_
#define AUTOWARE__NDT_SCAN_MATCHER__NDT_OMP__ESTIMATE_COVARIANCE_HPP_

#include "multigrid_ndt_omp.h"

#include <Eigen/Core>

#include <memory>
#include <utility>
#include <vector>

namespace pclomp
{

struct NdtCovarianceOptions {
  double information_scale = 1.0;
  double min_information_eigenvalue = 1e-6;
  double max_information_eigenvalue = 1e8;
  double min_translation_std = 0.05;
  double max_translation_std = 10.0;
  double min_rotation_std_rad = 0.5 * 3.14159265358979323846 / 180.0;
  double max_rotation_std_rad = 30.0 * 3.14159265358979323846 / 180.0;
};

struct ResultOfMultiNdtCovarianceEstimation
{
  Eigen::Vector2d mean = Eigen::Vector2d::Zero();
  Eigen::Matrix2d covariance = Eigen::Matrix2d::Identity();
  std::vector<Eigen::Matrix4f> ndt_initial_poses;
  std::vector<NdtResult> ndt_results;
  // 六维顺序与NDT的Hessian一致：[tx, ty, tz, rx, ry, rz]，角度单位为弧度。
  Eigen::Matrix<double, 6, 6> pose_covariance = Eigen::Matrix<double, 6, 6>::Identity();
  bool covariance_valid = false;
  bool degenerate = false;
};

/** \brief Estimate functions */
ResultOfMultiNdtCovarianceEstimation estimate_pose_covariance_by_laplace_approximation(
  const NdtResult & ndt_result, const NdtCovarianceOptions & options = NdtCovarianceOptions());
Eigen::Matrix2d estimate_xy_covariance_by_laplace_approximation(
  const Eigen::Matrix<double, 6, 6> & hessian);
ResultOfMultiNdtCovarianceEstimation estimate_xy_covariance_by_multi_ndt(
  const NdtResult & ndt_result,
  const std::shared_ptr<
    pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>> & ndt_ptr,
  const std::vector<Eigen::Matrix4f> & poses_to_search);
ResultOfMultiNdtCovarianceEstimation estimate_xy_covariance_by_multi_ndt_score(
  const NdtResult & ndt_result,
  const std::shared_ptr<
    pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>> & ndt_ptr,
  const std::vector<Eigen::Matrix4f> & poses_to_search, const double temperature);

/** \brief Propose poses to search.
 * (1) Compute covariance by Laplace approximation
 * (2) Find rotation matrix aligning covariance to principal axes
 * (3) Propose search points by adding offset_x and offset_y to the center_pose
 */
std::vector<Eigen::Matrix4f> propose_poses_to_search(
  const NdtResult & ndt_result, const std::vector<double> & offset_x,
  const std::vector<double> & offset_y);

/** \brief Calculate weights by exponential */
std::vector<double> calc_weight_vec(const std::vector<double> & score_vec, double temperature);

/** \brief Calculate weighted mean and covariance */
std::pair<Eigen::Vector2d, Eigen::Matrix2d> calculate_weighted_mean_and_cov(
  const std::vector<Eigen::Vector2d> & pose_2d_vec, const std::vector<double> & weight_vec);

/** \brief Rotate covariance to base_link coordinate */
Eigen::Matrix2d rotate_covariance_to_base_link(
  const Eigen::Matrix2d & covariance, const Eigen::Matrix4f & pose);

/** \brief Rotate covariance to map coordinate */
Eigen::Matrix2d rotate_covariance_to_map(
  const Eigen::Matrix2d & covariance, const Eigen::Matrix4f & pose);

/** \brief  Adjust diagonal covariance */
Eigen::Matrix2d adjust_diagonal_covariance(
  const Eigen::Matrix2d & covariance, const Eigen::Matrix4f & pose, const double fixed_cov00,
  const double fixed_cov11);

}  // namespace pclomp

#endif  // AUTOWARE__NDT_SCAN_MATCHER__NDT_OMP__ESTIMATE_COVARIANCE_HPP_
