#include "core/mapping/feature_extraction.h"
#include "core/mapping/mapping.h"
#include <pcl/common/transforms.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>

using namespace lightning;
void Require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }

// 16线射线与竖直平面、V形交角或遮挡台阶求交，intensity保存稳定点ID。
Scan Rays(int shape) {
    Scan scan; scan.stamp = 1; scan.cloud.reset(new PointCloudType);
    for (int ring = 0; ring < 16; ++ring) {
        const double elevation = (ring - 7.5) * 2 * std::acos(-1.0) / 180;
        for (int sample = 0; sample <= 240; ++sample) {
            const double angle = (sample - 120) * .005;
            if (shape == 3 && std::abs(angle) < .15) continue;
            double distance = 8.0 / (std::cos(elevation) * std::cos(angle));
            if (shape == 1) distance = 8.0 / (std::cos(elevation) * (std::cos(angle) + std::abs(std::sin(angle))));
            if ((shape == 2 || shape == 3) && angle >= 0) distance *= 2;
            if (shape == 4 && sample == 120) distance *= .5;
            PointType point;
            point.x = distance * std::cos(elevation) * std::cos(angle);
            point.y = distance * std::cos(elevation) * std::sin(angle);
            point.z = distance * std::sin(elevation);
            point.intensity = ring * 241 + sample;
            scan.cloud->push_back(point); scan.rings.push_back(ring);
        }
    }
    return scan;
}
std::set<int> Ids(const CloudPtr& cloud) {
    std::set<int> ids;
    for (const auto& point : cloud->points) ids.insert(static_cast<int>(point.intensity));
    return ids;
}
CloudPtr Clusters(double offset) {
    CloudPtr cloud(new PointCloudType);
    std::mt19937 random(42); std::normal_distribution<double> noise(0, .07);
    for (int group = 0; group < 12; ++group) {
        const Vec3d center(offset + 2.3 * (group % 4) + .25, 2.1 * (group / 4) + .27, .4 * (group % 3) + .23);
        for (int i = 0; i < 60; ++i) {
            const Vec3d delta(noise(random), noise(random), noise(random));
            for (double sign : {-1.0, 1.0}) {
                const Vec3d p = center + sign * delta;
                PointType point; point.x = p.x(); point.y = p.y(); point.z = p.z(); point.intensity = group;
                cloud->push_back(point);
            }
        }
    }
    return cloud;
}

int main() {
    try {
        FeatureOptions features; features.enabled = true;
        const auto plane = Rays(0), edge = Rays(1);
        const auto flat = ExtractFeatures(plane, plane.cloud, features);
        const auto selected = ExtractFeatures(edge, edge.cloud, features);
        Require(flat.corner->empty() && flat.surface->size() > 1000, "flat plane classification");
        Require(selected.corner->size() >= 16 && selected.surface->size() > 1000, "geometric edge not detected");
        const auto corners = Ids(selected.corner), surfaces = Ids(selected.surface);
        for (int id : corners) Require(!surfaces.count(id), "corner and surface overlap");

        // 任意输入顺序和ring交错，特征ID保持一致。
        std::vector<std::size_t> order(edge.cloud->size()); std::iota(order.begin(), order.end(), 0);
        std::mt19937 random(7); std::shuffle(order.begin(), order.end(), random);
        Scan shuffled; shuffled.cloud.reset(new PointCloudType);
        for (auto id : order) { shuffled.cloud->push_back(edge.cloud->points[id]); shuffled.rings.push_back(edge.rings[id]); }
        const auto reordered = ExtractFeatures(shuffled, shuffled.cloud, features);
        Require(Ids(reordered.corner) == corners && Ids(reordered.surface) == surfaces, "ring grouping/azimuth order depends on input layout");
        Scan echoes = edge; echoes.cloud.reset(new PointCloudType(*edge.cloud));
        for (std::size_t i = 0; i < edge.cloud->size(); ++i) {
            auto p = edge.cloud->points[i];
            const double angle = -2e-7;
            const double x = p.x, y = p.y;
            p.x = 2 * (std::cos(angle) * x - std::sin(angle) * y);
            p.y = 2 * (std::sin(angle) * x + std::cos(angle) * y); p.z *= 2; p.intensity += 10000;
            echoes.cloud->push_back(p); echoes.rings.push_back(edge.rings[i]);
        }
        const auto nearest_echo = ExtractFeatures(echoes, echoes.cloud, features);
        Require(Ids(nearest_echo.corner) == corners && Ids(nearest_echo.surface) == surfaces, "same-direction echoes did not retain near points");
        auto strict = features; strict.highpass_threshold = 1.0;
        Require(ExtractFeatures(edge, edge.cloud, strict).corner->empty(), "highpass threshold has no effect");
        strict = features; strict.lowpass_threshold = 0.0;
        Require(ExtractFeatures(plane, plane.cloud, strict).surface->empty(), "lowpass threshold has no effect");
        strict = features; strict.highpass_half_window = 1;
        Require(ExtractFeatures(edge, edge.cloud, strict).corner->size() < selected.corner->size(), "highpass window has no effect");
        strict = features; strict.lowpass_half_window = 1;
        Require(ExtractFeatures(plane, plane.cloud, strict).surface->size() > flat.surface->size(), "lowpass window has no effect");
        CloudPtr deskewed(new PointCloudType(*edge.cloud));
        for (auto& p : deskewed->points) { p.x += 1; p.y += 2; }
        const auto moved_selection = ExtractFeatures(edge, deskewed, features);
        Require(Ids(moved_selection.corner) == corners, "deskewing changed raw-depth feature labels");
        for (const auto& p : moved_selection.corner->points) {
            const auto& raw = edge.cloud->points[static_cast<std::size_t>(p.intensity)];
            Require(std::abs(p.x - raw.x - 1) < 1e-5 && std::abs(p.y - raw.y - 2) < 1e-5, "filtered depth moved point geometry");
        }
        auto gap = Rays(3);
        Require(ExtractFeatures(gap, gap.cloud, features).corner->empty(), "filter crosses angular gap");
        auto occluded = Rays(2);
        const auto boundary = ExtractFeatures(occluded, occluded.cloud, features);
        Require(!boundary.corner->empty(), "near-side occlusion edges were all discarded");
        for (const auto& point : boundary.corner->points) Require(point.x < 12, "occluded far-side edge classified as corner");
        auto spike = Rays(4);
        Require(ExtractFeatures(spike, spike.cloud, features).corner->empty(), "isolated depth spike classified as corner");
        auto missing = edge; missing.rings.clear();
        bool rejected = false;
        try { ExtractFeatures(missing, missing.cloud, features); } catch (const std::invalid_argument&) { rejected = true; }
        Require(rejected, "feature mode accepted missing ring metadata");

        NdtOptions options; options.num_layers = 2; options.num_threads = 1;
        options.resolutions = {1.6, .8, .4}; options.weights = {1, 2, 3};
        FeatureClouds target{Clusters(0), Clusters(25)};
        MultiLayerNdt joint(options); joint.SetFeatureTarget(target);
        MultiLayerNdt corner(options), surface(options); corner.SetTarget(target.corner); surface.SetTarget(target.surface);
        Vec6d pose; pose << .01, -.01, .005, .0003, -.0002, .0004;
        Vec6d g, gc, gs; Mat6d h, hc, hs; std::vector<double> scores, unused;
        const double value = joint.EvaluateFeatures(target, pose, g, h, scores);
        const double c = corner.Evaluate(target.corner, pose, gc, hc, unused);
        const double s = surface.Evaluate(target.surface, pose, gs, hs, unused);
        Require(scores.size() == 4 && joint.VoxelCounts().size() == 4, "feature mode does not have two independent pyramids");
        Require(std::abs(value - .5 * (c + s)) < 1e-10 && (g - .5 * (gc + gs)).norm() < 1e-8 &&
            (h - .5 * (hc + hs)).norm() < 1e-7, "feature score/gradient/Hessian not jointly weighted");
        for (auto count : joint.VoxelCounts()) Require(count > 0, "feature Gaussian pyramid is empty");
        FeatureClouds swapped{target.surface, target.corner};
        Require(joint.EvaluateFeatures(swapped, Vec6d::Zero(), gc, hc, unused) == 0.0,
            "corner and surface can match the wrong target category");
        FeatureClouds surface_only{CloudPtr(new PointCloudType), target.surface};
        Require(std::abs(joint.EvaluateFeatures(surface_only, pose, gc, hc, unused) - .5 * s) < 1e-10,
            "missing corner points blocked valid surface observations");
        for (int axis = 0; axis < 6; ++axis) {
            Vec6d plus = pose, minus = pose, gp, gm; Mat6d hp, hm;
            const double step = 2e-4; plus(axis) += step; minus(axis) -= step;
            const double fp = joint.EvaluateFeatures(target, plus, gp, hp, unused);
            const double fm = joint.EvaluateFeatures(target, minus, gm, hm, unused);
            Require(std::abs((fp - fm) / (2 * step) - g(axis)) < .03 * (1 + std::abs(g(axis))), "feature gradient finite difference");
            Require(((gp - gm) / (2 * step) - h.col(axis)).norm() < .03 * (1 + h.col(axis).norm()), "feature Hessian finite difference");
        }
        const SE3 truth = MultiLayerNdt::ToPose((Vec6d() << .04, -.025, .015, .0004, -.0003, .0006).finished());
        FeatureClouds source{CloudPtr(new PointCloudType), CloudPtr(new PointCloudType)};
        const Eigen::Matrix4f inverse = truth.inverse().matrix().cast<float>();
        pcl::transformPointCloud(*target.corner, *source.corner, inverse);
        pcl::transformPointCloud(*target.surface, *source.surface, inverse);
        const auto fit = joint.AlignFeatures(source, SE3());
        Require(fit.valid && (truth.inverse() * fit.pose).translation().norm() < .02 &&
            (truth.inverse() * fit.pose).so3().log().norm() < .003, "joint feature pose optimization failed");
        double expected = 0;
        for (int layer = 0; layer < 2; ++layer) {
            auto single = options; single.num_layers = 1;
            single.resolutions = {options.resolutions[layer]}; single.weights = {1};
            MultiLayerNdt one(single); one.SetTarget(target.corner);
            expected += options.weights[layer] / 3.0 * one.Evaluate(target.corner, pose, gc, hc, unused);
        }
        Require(std::abs(c - expected) < 1e-10, "num_layers does not select and normalize the first N levels");
        joint.SetTarget(target.corner);
        Require(joint.VoxelCounts().size() == 2 && std::abs(joint.Evaluate(target.corner, pose, gc, hc, unused) - c) < 1e-10,
            "full-cloud mode retained feature pyramids or weights");

        MappingOptions mapping_options; mapping_options.loop.loopClosureEnableFlag = false;
        mapping_options.features = features; mapping_options.ndt.num_layers = 2; mapping_options.keyframe_voxel = .05;
        Mapping mapping(mapping_options);
        const auto first = mapping.Process(edge);
        Require(first.valid && first.keyframe_added && first.layer_voxels.size() == 4 && first.corner_count > 0 && first.surface_count > 0,
            "feature mapping initialization chain");
        const auto& keyframe = mapping.Data().keyframes.front();
        Require(keyframe.features.corner && keyframe.features.surface && !keyframe.cloud->empty(), "keyframe feature/full clouds not retained separately");
        auto next = edge; next.stamp = 1.1;
        const auto second = mapping.Process(next);
        Require(second.valid && second.layer_scores.size() == 4, "feature matching chain did not run");
        mapping_options.features.enabled = false;
        Mapping full(mapping_options); missing.stamp = 1;
        const auto original = full.Process(missing);
        Require(original.valid && original.layer_voxels.size() == 2 && !full.Data().keyframes.front().features.corner &&
            !full.Data().keyframes.front().features.surface, "disabled mode allocates feature pyramids or needs rings");
        std::cout << "Features passed: plane surface=" << flat.surface->size() << ", corner=" << selected.corner->size()
                  << "; ring ordering, filters, class isolation, joint derivatives and both mapping modes\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
