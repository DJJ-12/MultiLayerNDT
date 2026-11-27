#include "core/mapping/mapping.h"
#include <pcl/common/transforms.h>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace lightning;
void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
CloudPtr Scene() {
    CloudPtr cloud(new PointCloudType);
    std::mt19937 random(42);
    std::normal_distribution<double> noise(0.0, .07);
    for (int group = 0; group < 12; ++group) {
        const Vec3d center(2.3 * (group % 4) + .25, 2.1 * (group / 4) + .27, .4 * (group % 3) + .23);
        for (int sample = 0; sample < 60; ++sample) {
            const Vec3d offset(noise(random), noise(random), noise(random));
            for (double sign : {-1.0, 1.0}) {
                const Vec3d xyz = center + sign * offset;
                PointType p; p.x = xyz.x(); p.y = xyz.y(); p.z = xyz.z(); p.intensity = group;
                cloud->push_back(p);
            }
        }
    }
    return cloud;
}

int main() {
    try {
        const auto target = Scene();
        NdtOptions options;
        options.resolutions = {1.6, .8, .4}; options.weights = {1, 2, 3};
        options.num_threads = 1; options.max_iterations = 60; options.epsilon = 1e-4;
        MultiLayerNdt joint(options); joint.SetTarget(target);
        Vec6d pose; pose << .01, -.01, .005, .003, -.002, .004;
        Vec6d gradient, expected_gradient = Vec6d::Zero();
        Mat6d hessian, expected_hessian = Mat6d::Zero();
        std::vector<double> scores;
        const double value = joint.Evaluate(target, pose, gradient, hessian, scores);
        double expected = 0;
        for (std::size_t layer = 0; layer < 3; ++layer) {
            auto single_options = options;
            single_options.num_layers = 1;
            single_options.resolutions = {options.resolutions[layer]}; single_options.weights = {1};
            MultiLayerNdt single(single_options); single.SetTarget(target);
            Vec6d g; Mat6d h; std::vector<double> s;
            const double weight = options.weights[layer] / 6;
            expected += weight * single.Evaluate(target, pose, g, h, s);
            expected_gradient += weight * g; expected_hessian += weight * h;
            Require(scores[layer] > 0, "a configured layer did not contribute");
        }
        Require(std::abs(value - expected) < 1e-10 && (gradient - expected_gradient).norm() < 1e-8 &&
            (hessian - expected_hessian).norm() < 1e-7, "joint score/gradient/Hessian does not equal weighted layers");
        for (int axis = 0; axis < 6; ++axis) {
            Vec6d plus = pose, minus = pose, gp, gm; Mat6d hp, hm; std::vector<double> unused;
            const double step = 2e-4;
            plus(axis) += step; minus(axis) -= step;
            const double fp = joint.Evaluate(target, plus, gp, hp, unused);
            const double fm = joint.Evaluate(target, minus, gm, hm, unused);
            Require(std::abs((fp - fm) / (2 * step) - gradient(axis)) < .03 * (1 + std::abs(gradient(axis))), "joint gradient finite difference");
            const Vec6d numeric_hessian = (gp - gm) / (2 * step);
            Require((numeric_hessian - hessian.col(axis)).norm() < .03 * (1 + hessian.col(axis).norm()), "joint Hessian finite difference");
        }
        const SE3 truth = MultiLayerNdt::ToPose((Vec6d() << .04, -.025, .015, .004, -.003, .006).finished());
        CloudPtr source(new PointCloudType);
        const Eigen::Matrix4f inverse = truth.inverse().matrix().cast<float>();
        pcl::transformPointCloud(*target, *source, inverse);
        const auto fit = joint.Align(source, SE3());
        Require(fit.valid && (truth.inverse() * fit.pose).translation().norm() < .04, "known pose registration failed");
        auto no_weight = options; no_weight.weights = {1, 0, 0};
        MultiLayerNdt coarse(no_weight); coarse.SetTarget(target);
        Vec6d g; Mat6d h; coarse.Evaluate(target, pose, g, h, scores);
        Require(scores[1] == 0 && scores[2] == 0, "zero-weight layers changed objective");
        MappingOptions mapping_options;
        mapping_options.loop.loopClosureEnableFlag = false;
        mapping_options.ndt = options; mapping_options.ndt.source_voxel = .02;
        mapping_options.keyframe_voxel = .02; mapping_options.keyframe_distance = .02;
        Mapping mapping(mapping_options);
        Scan first; first.stamp = 1; first.cloud = target;
        const auto initial = mapping.Process(first);
        Require(initial.valid && initial.keyframe_added && initial.state.pose.translation().norm() == 0, "first scan must establish map origin");
        Scan second; second.stamp = 1.1; second.cloud = source;
        Require(mapping.Process(second).valid && mapping.Data().trajectory.size() == 2, "mapping did not advance");
        Require(!mapping.Process(second).valid && mapping.Data().trajectory.size() == 2, "duplicate timestamp changed map");
        const auto destination = std::filesystem::temp_directory_path() /
            ("multi_layer_ndt_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        Require(Mapping::SaveMap(mapping.Data(), destination.string(), .05) &&
            std::filesystem::exists(destination / "global.pcd") && std::filesystem::file_size(destination / "trajectory.txt") > 0,
            "map/trajectory save failed");
        std::cout << "All layers, analytic derivatives, registration, mapping and save tests passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
