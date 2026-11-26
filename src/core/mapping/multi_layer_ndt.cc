#include "core/mapping/multi_layer_ndt.h"

#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <pcl/common/transforms.h>

namespace lightning {
class MultiLayerNdt::Layer : public pclomp::MultiGridNormalDistributionsTransform<PointType, PointType> {
public:
    Layer(const NdtOptions& options, double resolution, double weight) : weight(weight) {
        pclomp::NdtParams params{};
        params.resolution = static_cast<float>(resolution);
        params.num_threads = options.num_threads;
        setParams(params);
        // 与原computeTransformation中的高斯拟合完全相同；每层保持自己的参数。
        const double c1 = 10 * (1 - outlier_ratio_);
        const double c2 = outlier_ratio_ / std::pow(resolution, 3);
        gauss_d3_ = -std::log(c2);
        gauss_d1_ = -std::log(c1 + c2) - gauss_d3_;
        gauss_d2_ = -2 * std::log((-std::log(c1 * std::exp(-0.5) + c2) - gauss_d3_) / gauss_d1_);
        amplitude = -gauss_d1_;
    }
    double Score(PointCloudType& transformed, Vec6d& pose, Vec6d& gradient, Mat6d& hessian, bool full) {
        return computeDerivatives(gradient, hessian, transformed, pose, full);
    }
    double weight, amplitude;
    int group = 0;  // 全量模式为0；特征模式0为corner，1为surface。
    std::size_t voxel_count = 0;
    bool usable = false;
};

MultiLayerNdt::MultiLayerNdt(NdtOptions options) : options_(std::move(options)) {
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
    layers_.clear();  // 原MultiGrid的addTarget是追加语义；重建避免旧子地图残留。
    AppendLayers(target, 0, 1.0);
}

void MultiLayerNdt::SetFeatureTarget(const FeatureClouds& target) {
    layers_.clear();
    // 两类各占一半权重，各自按源点数归一化，避免surface数量淹没corner。
    AppendLayers(target.corner, 0, 0.5);
    AppendLayers(target.surface, 1, 0.5);
}

void MultiLayerNdt::AppendLayers(const CloudPtr& target, int group, double group_weight) {
    for (std::size_t i = 0; i < options_.resolutions.size(); ++i) {
        auto layer = std::make_unique<Layer>(options_, options_.resolutions[i], options_.weights[i] * group_weight);
        layer->group = group;
        if (options_.weights[i] > 0.0 && target && !target->empty()) {
            layer->setInputTarget(target);
            layer->voxel_count = layer->getVoxelPCD().size();
            layer->usable = layer->voxel_count > 0;
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

double MultiLayerNdt::EvaluateSources(const Sources& sources, const Vec6d& pose, Vec6d& gradient,
                                    Mat6d& hessian, std::vector<double>& scores, bool full) {
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
    double score = EvaluateSources(sources, pose, gradient, hessian, fit.layer_scores);
    if (!(score > 0) || !std::isfinite(score)) return fit;
    fit.valid = true;
    for (int iteration = 0; iteration < options_.max_iterations; ++iteration) {
        if (!gradient.allFinite() || !hessian.allFinite()) break;
        Eigen::JacobiSVD<Mat6d> solver(hessian, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Vec6d delta = solver.solve(-gradient);
        if (!delta.allFinite()) break;
        if (delta.norm() < options_.epsilon) { fit.converged = true; break; }
        if (gradient.dot(delta) <= 0) delta = gradient;  // 非极大方向改用上升方向。
        if (delta.norm() > options_.max_step) delta *= options_.max_step / delta.norm();
        bool advanced = false;
        for (int attempt = 0; attempt < 8; ++attempt, delta *= 0.5) {
            Vec6d trial_gradient;
            Mat6d trial_hessian;
            std::vector<double> trial_scores;
            const double trial_score = EvaluateSources(sources, pose + delta, trial_gradient, trial_hessian, trial_scores);
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
    return fit;
}
}  // namespace lightning
