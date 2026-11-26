#pragma once

#include "common/struct.h"
#include "GlobalLocalizer/ndt_omp/multigrid_ndt_omp.h"

namespace lightning {

// 各层共用一个位姿；复用原NDT的体素构建、对应查询和一/二阶导数。
class MultiLayerNdt {
public:
    explicit MultiLayerNdt(NdtOptions options);
    ~MultiLayerNdt();
    void SetTarget(const CloudPtr& target);
    NdtFit Align(const CloudPtr& source, const SE3& prediction);

    // 在给定的同一位姿上评价联合目标，测试也使用此接口核对解析导数。
    double Evaluate(const CloudPtr& source, const Vec6d& pose, Vec6d& gradient,
                    Mat6d& hessian, std::vector<double>& scores, bool compute_hessian = true);
    static SE3 ToPose(const Vec6d& parameters);
    static Vec6d ToParameters(const SE3& pose);

private:
    class Layer;
    NdtOptions options_;
    std::vector<std::unique_ptr<Layer>> layers_;
};
}  // namespace lightning
