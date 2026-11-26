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

#include "GlobalLocalizer/ndt_omp/estimate_covariance.hpp"

#include <algorithm>
#include <cmath>
#include <Eigen/Eigenvalues>
#include <fstream>
#include <iomanip>
#include <memory>
#include <utility>
#include <vector>

namespace pclomp
{

ResultOfMultiNdtCovarianceEstimation estimate_pose_covariance_by_laplace_approximation(
  const NdtResult & ndt_result, const NdtCovarianceOptions & options)
{
  ResultOfMultiNdtCovarianceEstimation result;
  const auto & hessian = ndt_result.hessian;
  if (!hessian.allFinite() || !ndt_result.pose.allFinite()) return result;
  result.mean = ndt_result.pose.topRightCorner<2, 1>().cast<double>();
  if (!std::isfinite(options.information_scale) || options.information_scale <= 0.0 ||
      !(0.0 < options.min_information_eigenvalue && options.min_information_eigenvalue <= options.max_information_eigenvalue) ||
      !(0.0 < options.min_translation_std && options.min_translation_std <= options.max_translation_std) ||
      !(0.0 < options.min_rotation_std_rad && options.min_rotation_std_rad <= options.max_rotation_std_rad)) return result;
  Eigen::Matrix<double, 6, 6> covariance;
  Eigen::Matrix<double, 6, 6> information = -0.5 * (hessian + hessian.transpose());
  information *= std::max(1e-12, options.information_scale);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(information);
  if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite()) return result;
  Eigen::Matrix<double, 6, 1> information_eigenvalues = solver.eigenvalues();
  // 必须在限幅前检查；数值正则化不能消除原始匹配的退化。
  result.degenerate = information_eigenvalues.minCoeff() <= options.min_information_eigenvalue;
  for (int i = 0; i < 6; ++i) information_eigenvalues(i) = std::clamp(information_eigenvalues(i), options.min_information_eigenvalue, options.max_information_eigenvalue);
  // 对特征值求倒数
  Eigen::Matrix<double, 6, 1> covariance_eigenvalues = information_eigenvalues.cwiseInverse();
  // 再把特征向量乘回去，然后得到新的方差
  covariance = solver.eigenvectors() * covariance_eigenvalues.asDiagonal() * solver.eigenvectors().transpose();
  covariance = (0.5 * (covariance + covariance.transpose())).eval();

  Eigen::Matrix<double, 6, 1> standard_deviation = covariance.diagonal().cwiseMax(1e-12).cwiseSqrt();
  for (int i = 0; i < 3; ++i) standard_deviation(i) = std::clamp(standard_deviation(i), options.min_translation_std, options.max_translation_std);
  for (int i = 3; i < 6; ++i) standard_deviation(i) = std::clamp(standard_deviation(i), options.min_rotation_std_rad, options.max_rotation_std_rad);
  Eigen::Matrix<double, 6, 6> correlation = Eigen::Matrix<double, 6, 6>::Zero();
  for (int row = 0; row < 6; ++row) {
    for (int col = 0; col < 6; ++col) {
      const double denominator = std::sqrt(std::max(1e-12, covariance(row, row) * covariance(col, col)));
      correlation(row, col) = std::clamp(covariance(row, col) / denominator, -0.99, 0.99);
    }
    correlation(row, row) = 1.0;
  }
  covariance = standard_deviation.asDiagonal() * correlation * standard_deviation.asDiagonal();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> covariance_solver(0.5 * (covariance + covariance.transpose()));
  if (covariance_solver.info() != Eigen::Success) return result;
  Eigen::Matrix<double, 6, 1> eigenvalues = covariance_solver.eigenvalues().cwiseMax(1e-12);
  covariance = covariance_solver.eigenvectors() * eigenvalues.asDiagonal() * covariance_solver.eigenvectors().transpose();
  if (!covariance.allFinite()) return result;
  result.pose_covariance = covariance;
  result.covariance = covariance.topLeftCorner<2, 2>();
  result.covariance_valid = !result.degenerate;
  return result;
}

Eigen::Matrix2d estimate_xy_covariance_by_laplace_approximation(
  const Eigen::Matrix<double, 6, 6> & hessian)
{
  const Eigen::Matrix2d hessian_xy = hessian.block<2, 2>(0, 0);
  Eigen::Matrix2d covariance_xy = -hessian_xy.inverse();
  return covariance_xy;
}

ResultOfMultiNdtCovarianceEstimation estimate_xy_covariance_by_multi_ndt(
  const NdtResult & ndt_result,
  const std::shared_ptr<
    pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>> & ndt_ptr,
  const std::vector<Eigen::Matrix4f> & poses_to_search)
{
  // initialize by the main result
  const Eigen::Vector2d ndt_pose_2d(ndt_result.pose(0, 3), ndt_result.pose(1, 3));
  std::vector<Eigen::Vector2d> ndt_pose_2d_vec{ndt_pose_2d};

  // multiple searches
  std::vector<NdtResult> ndt_results;
  for (const Eigen::Matrix4f & curr_pose : poses_to_search) {
    auto sub_output_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    ndt_ptr->align(*sub_output_cloud, curr_pose);
    const NdtResult sub_ndt_result = ndt_ptr->getResult();
    ndt_results.push_back(sub_ndt_result);

    const Eigen::Matrix4f sub_ndt_pose = sub_ndt_result.pose;
    const Eigen::Vector2d sub_ndt_pose_2d = sub_ndt_pose.topRightCorner<2, 1>().cast<double>();
    ndt_pose_2d_vec.emplace_back(sub_ndt_pose_2d);
  }

  // calculate the weights
  const int n = static_cast<int>(ndt_results.size()) + 1;
  const std::vector<double> weight_vec(n, 1.0 / n);

  // calculate mean and covariance
  std::pair<Eigen::Vector2d, Eigen::Matrix2d> mean_cov_result = calculate_weighted_mean_and_cov(ndt_pose_2d_vec, weight_vec);
  Eigen::Vector2d mean = mean_cov_result.first;
  Eigen::Matrix2d covariance = mean_cov_result.second;

  // unbiased covariance
  covariance *= static_cast<double>(n - 1) / n;

  return {mean, covariance, poses_to_search, ndt_results};
}

ResultOfMultiNdtCovarianceEstimation estimate_xy_covariance_by_multi_ndt_score(
  const NdtResult & ndt_result,
  const std::shared_ptr<
    pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>> & ndt_ptr,
  const std::vector<Eigen::Matrix4f> & poses_to_search, const double temperature)
{
  // initialize by the main result
  const Eigen::Vector2d ndt_pose_2d(ndt_result.pose(0, 3), ndt_result.pose(1, 3));
  std::vector<Eigen::Vector2d> ndt_pose_2d_vec{ndt_pose_2d};
  std::vector<double> score_vec{ndt_result.nearest_voxel_transformation_likelihood};

  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr input_cloud = ndt_ptr->getInputCloud();
  pcl::PointCloud<pcl::PointXYZ> trans_cloud;

  // multiple searches
  std::vector<NdtResult> ndt_results;
  for (const Eigen::Matrix4f & curr_pose : poses_to_search) {
    const Eigen::Vector2d sub_ndt_pose_2d = curr_pose.topRightCorner<2, 1>().cast<double>();
    ndt_pose_2d_vec.emplace_back(sub_ndt_pose_2d);

    transformPointCloud(*input_cloud, trans_cloud, curr_pose);
    const double nvtl = ndt_ptr->calculateNearestVoxelTransformationLikelihood(trans_cloud);
    score_vec.emplace_back(nvtl);

    NdtResult sub_ndt_result{};
    sub_ndt_result.pose = curr_pose;
    sub_ndt_result.iteration_num = 0;
    sub_ndt_result.nearest_voxel_transformation_likelihood = static_cast<float>(nvtl);
    ndt_results.push_back(sub_ndt_result);
  }

  // calculate the weights
  const std::vector<double> weight_vec = calc_weight_vec(score_vec, temperature);

  // calculate mean and covariance
  const std::pair<Eigen::Vector2d, Eigen::Matrix2d> mean_cov_result = calculate_weighted_mean_and_cov(ndt_pose_2d_vec, weight_vec);
  const Eigen::Vector2d mean = mean_cov_result.first;
  const Eigen::Matrix2d covariance = mean_cov_result.second;
  return {mean, covariance, poses_to_search, ndt_results};
}

std::vector<Eigen::Matrix4f> propose_poses_to_search(
  const NdtResult & ndt_result, const std::vector<double> & offset_x,
  const std::vector<double> & offset_y)
{
  assert(offset_x.size() == offset_y.size());
  const Eigen::Matrix4f & center_pose = ndt_result.pose;
  const Eigen::Matrix2d rot = ndt_result.pose.topLeftCorner<2, 2>().cast<double>();
  std::vector<Eigen::Matrix4f> poses_to_search;
  for (int i = 0; i < static_cast<int>(offset_x.size()); i++) {
    const Eigen::Vector2d pose_offset(offset_x[i], offset_y[i]);
    const Eigen::Vector2d rotated_pose_offset_2d = rot * pose_offset;
    Eigen::Matrix4f curr_pose = center_pose;
    curr_pose(0, 3) += static_cast<float>(rotated_pose_offset_2d.x());
    curr_pose(1, 3) += static_cast<float>(rotated_pose_offset_2d.y());
    poses_to_search.emplace_back(curr_pose);
  }
  return poses_to_search;
}

std::vector<double> calc_weight_vec(const std::vector<double> & score_vec, double temperature)
{
  const int n = static_cast<int>(score_vec.size());
  const double max_score = *std::max_element(score_vec.begin(), score_vec.end());
  std::vector<double> exp_score_vec(n);
  double exp_score_sum = 0.0;
  for (int i = 0; i < n; i++) {
    exp_score_vec[i] = std::exp((score_vec[i] - max_score) / temperature);
    exp_score_sum += exp_score_vec[i];
  }
  for (int i = 0; i < n; i++) {
    exp_score_vec[i] /= exp_score_sum;
  }
  return exp_score_vec;
}

std::pair<Eigen::Vector2d, Eigen::Matrix2d> calculate_weighted_mean_and_cov(
  const std::vector<Eigen::Vector2d> & pose_2d_vec, const std::vector<double> & weight_vec)
{
  const int n = static_cast<int>(pose_2d_vec.size());
  Eigen::Vector2d mean = Eigen::Vector2d::Zero();
  for (int i = 0; i < n; i++) {
    mean += weight_vec[i] * pose_2d_vec[i];
  }
  Eigen::Matrix2d covariance = Eigen::Matrix2d::Zero();
  for (int i = 0; i < n; i++) {
    const Eigen::Vector2d diff = pose_2d_vec[i] - mean;
    covariance += weight_vec[i] * diff * diff.transpose();
  }
  return {mean, covariance};
}

Eigen::Matrix2d rotate_covariance_to_base_link(
  const Eigen::Matrix2d & covariance, const Eigen::Matrix4f & pose)
{
  const Eigen::Matrix2d rot = pose.topLeftCorner<2, 2>().cast<double>();
  return rot.transpose() * covariance * rot;
}

Eigen::Matrix2d rotate_covariance_to_map(
  const Eigen::Matrix2d & covariance, const Eigen::Matrix4f & pose)
{
  const Eigen::Matrix2d rot = pose.topLeftCorner<2, 2>().cast<double>();
  return rot * covariance * rot.transpose();
}

Eigen::Matrix2d adjust_diagonal_covariance(
  const Eigen::Matrix2d & covariance, const Eigen::Matrix4f & pose, const double fixed_cov00,
  const double fixed_cov11)
{
  Eigen::Matrix2d cov_base_link = rotate_covariance_to_base_link(covariance, pose);
  cov_base_link(0, 0) = std::max(cov_base_link(0, 0), fixed_cov00);
  cov_base_link(1, 1) = std::max(cov_base_link(1, 1), fixed_cov11);
  Eigen::Matrix2d cov_map = rotate_covariance_to_map(cov_base_link, pose);
  return cov_map;
}

}  // namespace pclomp
