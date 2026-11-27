#include "core/mapping/multi_layer_ndt.h"

#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

namespace lightning {
class MultiLayerNdt::Layer : public pclomp::MultiGridNormalDistributionsTransform<PointType, PointType> {
public:
    Layer(const NdtOptions& options, double resolution, double weight, bool geometry)
        : weight(weight), resolution(resolution) {
        pclomp::NdtParams params{};
        params.resolution = static_cast<float>(resolution);
        params.num_threads = options.num_threads;
        setParams(params);
        if (geometry) {
            // exp(-e'Qe/2)：Q已经包含线/面投影和1/h²，复用原概率导数。
            gauss_d1_ = -1.0;
            gauss_d2_ = 1.0;
        } else {
            // 全量模式保持原NDT高斯拟合参数。
            const double c1 = 10 * (1 - outlier_ratio_);
            const double c2 = outlier_ratio_ / std::pow(resolution, 3);
            gauss_d3_ = -std::log(c2);
            gauss_d1_ = -std::log(c1 + c2) - gauss_d3_;
            gauss_d2_ = -2 * std::log((-std::log(c1 * std::exp(-0.5) + c2) - gauss_d3_) / gauss_d1_);
        }
        amplitude = -gauss_d1_;
    }
    void SetTarget(const CloudPtr& target, bool geometry) {
        if (geometry) {
            // 几何拟合只需要占用体素质心，不套用NDT每格至少6点的限制。
            centroids.reset(new PointCloudType);
            pcl::VoxelGrid<PointType> filter;
            filter.setLeafSize(resolution, resolution, resolution);
            filter.setInputCloud(target);
            filter.filter(*centroids);
            voxel_count = centroids->size();
            if (voxel_count) {
                tree = std::make_unique<pcl::KdTreeFLANN<PointType>>();
                tree->setInputCloud(centroids);
            }
        } else {
            setInputTarget(target);
            voxel_count = getVoxelPCD().size();
        }
        usable = voxel_count > 0;
    }
    double Score(PointCloudType& transformed, Vec6d& pose, Vec6d& gradient, Mat6d& hessian, bool full) {
        return computeDerivatives(gradient, hessian, transformed, pose, full);
    }
    double ScoreGeometry(const std::vector<FeatureCorrespondence>& correspondences, Vec6d& pose,
                         Vec6d& gradient, Mat6d& hessian, std::vector<double>& scores, bool full, bool gauss_newton) {
        gradient.setZero(); hessian.setZero(); std::fill(scores.begin(), scores.end(), 0.0);
        const bool exact_hessian = full && !gauss_newton;
        computeAngleDerivatives(pose, exact_hessian);
        const SE3 transform = MultiLayerNdt::ToPose(pose);
        double total = 0.0;
#pragma omp parallel num_threads(params_.num_threads)
        {
            double sum = 0.0;
            Vec6d g = Vec6d::Zero(); Mat6d h = Mat6d::Zero();
            std::vector<double> parts(scores.size(), 0.0);
            Eigen::Matrix<double, 4, 6> point_gradient = Eigen::Matrix<double, 4, 6>::Zero();
            Eigen::Matrix<double, 24, 6> point_hessian = Eigen::Matrix<double, 24, 6>::Zero();
            point_gradient.topLeftCorner<3, 3>().setIdentity();
#pragma omp for schedule(static)
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(correspondences.size()); ++i) {
                const auto& c = correspondences[i];
                if (c.layer_weights.empty()) continue;
                Vec3d point = c.point;
                computePointDerivatives(point, point_gradient, point_hessian, exact_hessian);
                const Vec3d error = transform * point - c.reference;
                const double value = updateDerivatives(g, h, point_gradient, point_hessian, error, c.metric, full, true, gauss_newton);
                sum += value;
                for (std::size_t layer = 0; layer < parts.size(); ++layer) parts[layer] += c.layer_weights[layer] * value;
            }
#pragma omp critical(multi_layer_feature_score)
            {
                total += sum; gradient += g; hessian += h;
                for (std::size_t layer = 0; layer < scores.size(); ++layer) scores[layer] += parts[layer];
            }
        }
        return total;
    }
    double weight, amplitude, resolution;
    int group = 0;  // 全量模式为0；特征模式0为corner，1为surface。
    std::size_t voxel_count = 0;
    bool usable = false;
    CloudPtr centroids;
    std::unique_ptr<pcl::KdTreeFLANN<PointType>> tree;
};

MultiLayerNdt::MultiLayerNdt(NdtOptions options, FeatureOptions features)
    : options_(std::move(options)), features_(std::move(features)) {
    if (options_.resolutions.empty() || options_.resolutions.size() != options_.weights.size())
        throw std::invalid_argument("ndt.resolutions与weights需要有相同的非零长度");
    if (options_.num_layers < 1 || options_.num_layers > static_cast<int>(options_.resolutions.size()))
        throw std::invalid_argument("ndt.num_layers必须在1与resolutions/weights的长度之间");
    options_.resolutions.resize(options_.num_layers);
    options_.weights.resize(options_.num_layers);
    const double total = std::accumulate(options_.weights.begin(), options_.weights.end(), 0.0);
    if (!(total > 0)) throw std::invalid_argument("NDT权重和必须大于0");
    for (std::size_t i = 0; i < options_.weights.size(); ++i) {
        if (!(options_.resolutions[i] > 0) || !(options_.weights[i] >= 0))
            throw std::invalid_argument("体素尺寸必须为正，权重不能为负");
        options_.weights[i] /= total;
    }
}
MultiLayerNdt::~MultiLayerNdt() = default;

void MultiLayerNdt::SetTarget(const CloudPtr& target) {
    features_.enabled = false;
    layers_.clear();  // 原MultiGrid的addTarget是追加语义；重建避免旧子地图残留。
    AppendLayers(target, 0, 1.0);
}

void MultiLayerNdt::SetFeatureTarget(const FeatureClouds& target) {
    features_.enabled = true;
    layers_.clear();
    // 两类各占一半权重，各自按源点数归一化，避免surface数量淹没corner。
    AppendLayers(target.corner, 0, 0.5);
    AppendLayers(target.surface, 1, 0.5);
}

void MultiLayerNdt::AppendLayers(const CloudPtr& target, int group, double group_weight) {
    for (std::size_t i = 0; i < options_.resolutions.size(); ++i) {
        auto layer = std::make_unique<Layer>(options_, options_.resolutions[i], options_.weights[i] * group_weight, features_.enabled);
        layer->group = group;
        if (options_.weights[i] > 0.0 && target && !target->empty()) {
            layer->SetTarget(target, features_.enabled);
        }
        layers_.push_back(std::move(layer));
    }
}

std::vector<std::size_t> MultiLayerNdt::VoxelCounts() const {
    std::vector<std::size_t> counts;
    for (const auto& layer : layers_) counts.push_back(layer->voxel_count);
    return counts;
}

SE3 MultiLayerNdt::ToPose(const Vec6d& p) {
    const Mat3d rotation = (Eigen::AngleAxisd(p(3), Vec3d::UnitX()) *
        Eigen::AngleAxisd(p(4), Vec3d::UnitY()) * Eigen::AngleAxisd(p(5), Vec3d::UnitZ())).toRotationMatrix();
    return SE3(rotation, p.head<3>());
}
Vec6d MultiLayerNdt::ToParameters(const SE3& pose) {
    Vec6d parameters;
    parameters.head<3>() = pose.translation();
    parameters.tail<3>() = pose.rotationMatrix().eulerAngles(0, 1, 2);
    return parameters;
}

double MultiLayerNdt::Evaluate(const CloudPtr& source, const Vec6d& pose, Vec6d& gradient,
                             Mat6d& hessian, std::vector<double>& scores, bool full) {
    return EvaluateSources({source, nullptr}, pose, gradient, hessian, scores, full);
}

double MultiLayerNdt::EvaluateFeatures(const FeatureClouds& source, const Vec6d& pose, Vec6d& gradient,
                                     Mat6d& hessian, std::vector<double>& scores, bool full) {
    return EvaluateSources({source.corner, source.surface}, pose, gradient, hessian, scores, full);
}

MultiLayerNdt::Correspondences MultiLayerNdt::FindCorrespondences(const Sources& sources, const Vec6d& pose) const {
    Correspondences result;
    const SE3 transform = ToPose(pose);
    const int levels = options_.num_layers;
    // 不同层共同拟合同一条线或同一个面，支持点必须来自同一尺度的局部邻域。
    double finest_resolution = std::numeric_limits<double>::infinity();
    for (int level = 0; level < levels; ++level) {
        if (options_.weights[level] > 0.0)
            finest_resolution = std::min(finest_resolution, options_.resolutions[level]);
    }
    const double radius = features_.neighbor_radius_factor * finest_resolution;
    const double radius_squared = radius * radius;
    for (int group = 0; group < 2; ++group) {
        if (!sources[group]) continue;
        result[group].resize(sources[group]->size());
#pragma omp parallel num_threads(options_.num_threads)
        {
            // 每线程复用搜索和QR缓冲，避免每个点反复申请动态矩阵。
            std::vector<int> ids(5);
            std::vector<float> distances(5);
            std::vector<Vec3d> centers; centers.reserve(5 * levels);
            std::vector<double> weights; weights.reserve(5 * levels);
            std::vector<double> layer_weights(levels);
            Eigen::MatrixXd A(5 * levels, 3);
#pragma omp for schedule(static)
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(sources[group]->size()); ++i) {
                const auto& p = sources[group]->points[i];
                const Vec3d point(p.x, p.y, p.z), query = transform * point;
                PointType search;
                search.x = static_cast<float>(query.x()); search.y = static_cast<float>(query.y()); search.z = static_cast<float>(query.z());
                centers.clear(); weights.clear(); std::fill(layer_weights.begin(), layer_weights.end(), 0.0);
                for (int level = 0; level < levels; ++level) {
                    const auto& layer = *layers_[group * levels + level];
                    if (!layer.usable || layer.weight == 0.0) continue;
                    int count = layer.tree->nearestKSearch(search, 5, ids, distances);
                    while (count && distances[count - 1] > radius_squared) --count;
                    if (!count) continue;
                    layer_weights[level] = layer.weight;
                    for (int j = 0; j < count; ++j) {
                        const auto& center = layer.centroids->points[ids[j]];
                        const Vec3d q(center.x, center.y, center.z);
                        // 跨层重合质心合并权重，不把同一点重复计作多个几何支持。
                        std::size_t k = 0;
                        while (k < centers.size() && (centers[k] - q).squaredNorm() > 1e-8) ++k;
                        if (k == centers.size()) { centers.push_back(q); weights.push_back(0.0); }
                        weights[k] += layer.weight / count;
                    }
                }
                if (centers.size() < 5) continue;
                const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
                Vec3d reference = Vec3d::Zero(); Mat3d projection;
                for (std::size_t j = 0; j < centers.size(); ++j) reference += weights[j] * centers[j] / total;
                if (group == 0) {
                    Mat3d covariance = Mat3d::Zero();
                    for (std::size_t j = 0; j < centers.size(); ++j) {
                        const Vec3d error = centers[j] - reference;
                        covariance.noalias() += (weights[j] / total) * error * error.transpose();
                    }
                    Eigen::SelfAdjointEigenSolver<Mat3d> eigen(covariance);
                    if (eigen.info() != Eigen::Success || eigen.eigenvalues()(2) <= 1e-12 ||
                        eigen.eigenvalues()(2) <= 3 * eigen.eigenvalues()(1)) continue;
                    const Vec3d direction = eigen.eigenvectors().col(2);
                    projection = Mat3d::Identity() - direction * direction.transpose();
                } else {
                    // 先减去质心，用列主元QR选出两个切向坐标，求第三个坐标对它们的线性关系。
                    // 平面经过质心。固定全局pd=1会漏掉经过map原点的平面。
                    const int count = static_cast<int>(centers.size());
                    for (int j = 0; j < count; ++j) {
                        A.row(j) = std::sqrt(weights[j]) * (centers[j] - reference).transpose();
                    }
                    const auto qr = A.topRows(count).colPivHouseholderQr();
                    if (qr.rank() < 2) continue;
                    const auto& R = qr.matrixR();
                    Vec3d coefficients;
                    coefficients.head<2>() = R.topLeftCorner<2, 2>().triangularView<Eigen::Upper>().solve(-R.block<2, 1>(0, 2));
                    coefficients.z() = 1.0;
                    const Vec3d normal = (qr.colsPermutation() * coefficients).normalized();
                    bool plane_valid = true;
                    for (const auto& q : centers) if (std::abs(normal.dot(q - reference)) > 0.2) { plane_valid = false; break; }
                    if (!plane_valid) continue;
                    projection = normal * normal.transpose();
                }
                auto& c = result[group][i];
                const double scale = group == 0 ? features_.corner_distance_scale : features_.surface_distance_scale;
                c.point = point; c.reference = reference; c.metric = projection / (scale * scale);
                c.layer_weights = layer_weights;
                for (auto& weight : c.layer_weights) weight /= total;
            }
        }
    }
    return result;
}

double MultiLayerNdt::EvaluateGeometry(const Sources& sources, const Correspondences& correspondences,
                                     const Vec6d& pose, Vec6d& gradient, Mat6d& hessian,
                                     std::vector<double>& scores, bool full, bool gauss_newton) {
    gradient.setZero(); hessian.setZero(); scores.assign(layers_.size(), 0.0);
    double score = 0.0;
    for (int group = 0; group < 2; ++group) {
        if (!sources[group] || sources[group]->empty()) continue;
        Vec6d g, p = pose; Mat6d h;
        std::vector<double> parts(options_.num_layers);
        // 跨层拟合后每个源点只计一次分数，不在原来的体素循环中重复添加。
        const double raw = layers_[group * options_.num_layers]->ScoreGeometry(correspondences[group], p, g, h, parts, full, gauss_newton);
        const double normalizer = 0.5 / sources[group]->size();
        score += normalizer * raw;
        gradient.noalias() += normalizer * g; hessian.noalias() += normalizer * h;
        for (int level = 0; level < options_.num_layers; ++level)
            scores[group * options_.num_layers + level] = parts[level] / sources[group]->size();
    }
    return score;
}

double MultiLayerNdt::EvaluateSources(const Sources& sources, const Vec6d& pose, Vec6d& gradient,
                                    Mat6d& hessian, std::vector<double>& scores, bool full, const Correspondences* fixed) {
    if (features_.enabled) {
        if (fixed) return EvaluateGeometry(sources, *fixed, pose, gradient, hessian, scores, full, !features_.exact_hessian);
        return EvaluateGeometry(sources, FindCorrespondences(sources, pose), pose, gradient, hessian, scores, full, false);
    }
    gradient.setZero(); hessian.setZero(); scores.assign(layers_.size(), 0.0);
    std::array<PointCloudType, 2> transformed;
    const Eigen::Matrix4f transform = ToPose(pose).matrix().cast<float>();
    for (std::size_t group = 0; group < sources.size(); ++group)
        if (sources[group] && !sources[group]->empty()) pcl::transformPointCloud(*sources[group], transformed[group], transform);
    double score = 0.0;
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        auto& layer = *layers_[i];
        const auto& source = sources[layer.group];
        if (!layer.usable || layer.weight == 0.0 || !source || source->empty()) continue;
        layer.setInputSource(source);
        Vec6d g, p = pose;
        Mat6d h;
        const double raw = layer.Score(transformed[layer.group], p, g, h, full);
        // 固定总点数与高斯幅值归一化；不使用随位姿变化的“成功匹配点数”作分母。
        const double normalizer = 1.0 / (source->size() * layer.amplitude);
        scores[i] = normalizer * raw;
        score += layer.weight * scores[i];
        gradient.noalias() += (layer.weight * normalizer) * g;
        hessian.noalias() += (layer.weight * normalizer) * h;
    }
    return score;
}

NdtFit MultiLayerNdt::Align(const CloudPtr& source, const SE3& prediction) {
    return AlignSources({source, nullptr}, prediction);
}

NdtFit MultiLayerNdt::AlignFeatures(const FeatureClouds& source, const SE3& prediction) {
    return AlignSources({source.corner, source.surface}, prediction);
}

NdtFit MultiLayerNdt::AlignSources(const Sources& sources, const SE3& prediction) {
    NdtFit fit;
    fit.pose = prediction;
    Vec6d pose = ToParameters(prediction), gradient;
    Mat6d hessian;
    Correspondences correspondences;
    if (features_.enabled) correspondences = FindCorrespondences(sources, pose);
    const Correspondences* fixed = features_.enabled ? &correspondences : nullptr;
    double score = EvaluateSources(sources, pose, gradient, hessian, fit.layer_scores, true, fixed);
    if (!(score > 0) || !std::isfinite(score)) return fit;
    fit.valid = true;
    for (int iteration = 0; iteration < options_.max_iterations; ++iteration) {
        if (features_.enabled && iteration > 0) {
            // 本轮回溯固定模型，接受位姿后下一轮才重新找对应和拟合。
            correspondences = FindCorrespondences(sources, pose);
            score = EvaluateSources(sources, pose, gradient, hessian, fit.layer_scores, true, fixed);
            if (!(score > 0) || !std::isfinite(score)) break;
        }
        if (!gradient.allFinite() || !hessian.allFinite()) break;
        Eigen::JacobiSVD<Mat6d> solver(hessian, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Vec6d delta = solver.solve(-gradient);
        if (!delta.allFinite()) break;
        if (delta.norm() < options_.epsilon) { fit.converged = true; break; }
        if (gradient.dot(delta) <= 0) {
            if (features_.enabled && features_.exact_hessian) {
                // 精确Hessian可能不定。Newton步非上升时，用同一Hessian的奇异值缩放梯度。
                const auto rank = solver.rank();
                Vec6d scaled = Vec6d::Zero();
                scaled.head(rank) = (solver.matrixV().leftCols(rank).transpose() * gradient)
                                        .cwiseQuotient(solver.singularValues().head(rank));
                delta = solver.matrixV() * scaled;
            } else {
                delta = gradient;
            }
        }
        if (delta.norm() > options_.max_step) delta *= options_.max_step / delta.norm();
        bool advanced = false;
        for (int attempt = 0; attempt < 8; ++attempt, delta *= 0.5) {
            Vec6d trial_gradient;
            Mat6d trial_hessian;
            std::vector<double> trial_scores;
            // 特征路径下一轮会重建对应和Hessian，回溯不重复计算这部分。
            const double trial_score = EvaluateSources(sources, pose + delta, trial_gradient, trial_hessian, trial_scores, !features_.enabled, fixed);
            if (!std::isfinite(trial_score) || trial_score < score) continue;
            pose += delta;
            score = trial_score; gradient = trial_gradient; hessian = trial_hessian;
            fit.layer_scores = std::move(trial_scores);
            ++fit.iterations;
            advanced = true;
            if (delta.norm() < options_.epsilon) fit.converged = true;
            break;
        }
        if (!advanced || fit.converged) break;
    }
    fit.pose = ToPose(pose);
    fit.score = score;
    if (features_.enabled) {
        const auto valid = [](const FeatureCorrespondence& c) { return !c.layer_weights.empty(); };
        fit.corner_matches = std::count_if(correspondences[0].begin(), correspondences[0].end(), valid);
        fit.surface_matches = std::count_if(correspondences[1].begin(), correspondences[1].end(), valid);
    }
    return fit;
}
}  // namespace lightning
