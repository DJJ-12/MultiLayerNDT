#pragma once

#include "common/struct.h"
#include "GlobalLocalizer/ndt_omp/multigrid_ndt_omp.h"
#include <array>

namespace lightning {

// 各层共用一个位姿；全量模式使用NDT，特征模式使用跨层线/面几何指数评分。
class MultiLayerNdt {
public:
    explicit MultiLayerNdt(NdtOptions options, FeatureOptions features = {});
    ~MultiLayerNdt();
    void SetTarget(const CloudPtr& target);
    void SetFeatureTarget(const FeatureClouds& target);
    NdtFit Align(const CloudPtr& source, const SE3& prediction);
    NdtFit AlignFeatures(const FeatureClouds& source, const SE3& prediction);
    std::vector<std::size_t> VoxelCounts() const;

    // 在给定的同一位姿上评价联合目标，测试也使用此接口核对解析导数。
    double Evaluate(const CloudPtr& source, const Vec6d& pose, Vec6d& gradient,
                    Mat6d& hessian, std::vector<double>& scores, bool compute_hessian = true);
    double EvaluateFeatures(const FeatureClouds& source, const Vec6d& pose, Vec6d& gradient,
                            Mat6d& hessian, std::vector<double>& scores, bool compute_hessian = true);
    static SE3 ToPose(const Vec6d& parameters);
    static Vec6d ToParameters(const SE3& pose);

private:
    class Layer;
    using Sources = std::array<CloudPtr, 2>;
    using Correspondences = std::array<std::vector<FeatureCorrespondence>, 2>;
    void AppendLayers(const CloudPtr& target, int group, double group_weight);
    Correspondences FindCorrespondences(const Sources& sources, const Vec6d& pose) const;
    double EvaluateGeometry(const Sources& sources, const Correspondences& correspondences,
                            const Vec6d& pose, Vec6d& gradient, Mat6d& hessian,
                            std::vector<double>& scores, bool compute_hessian, bool gauss_newton,
                            Mat6d* information = nullptr);
    double EvaluateSources(const Sources& sources, const Vec6d& pose, Vec6d& gradient,
                           Mat6d& hessian, std::vector<double>& scores, bool compute_hessian = true,
                           const Correspondences* fixed = nullptr, Mat6d* information = nullptr);
    NdtFit AlignSources(const Sources& sources, const SE3& prediction);
    NdtOptions options_;
    FeatureOptions features_;
    std::vector<std::unique_ptr<Layer>> layers_;
};
}  // namespace lightning
