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
            if (shape == 1) distance = 8.0 / (std::cos(elevation) * (std::cos(angle - .08) + std::abs(std::sin(angle - .08))));
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

FeatureClouds GeometryScene() {
    FeatureClouds clouds{CloudPtr(new PointCloudType), CloudPtr(new PointCloudType)};
    auto add = [](const CloudPtr& cloud, const Vec3d& v) {
        PointType p; p.x = v.x(); p.y = v.y(); p.z = v.z(); cloud->push_back(p);
    };
    for (int i = -16; i <= 16; ++i) {
        add(clouds.corner, Vec3d(.13 + .19 * i, 2.17, 3.13));
        add(clouds.corner, Vec3d(12.17, .13 + .19 * i, 5.27));
        add(clouds.corner, Vec3d(-12.27, 4.13, .17 + .19 * i));
    }
    for (int i = -5; i <= 5; ++i) for (int j = -5; j <= 5; ++j) {
        add(clouds.surface, Vec3d(25.13 + .22 * i, .17 + .22 * j, 4.13));
        add(clouds.surface, Vec3d(-25.13, .19 + .22 * i, 4.17 + .22 * j));
        add(clouds.surface, Vec3d(.21 + .22 * i, 25.23, 3.17 + .22 * j));
    }
    return clouds;
}

int main() {
    try {
        FeatureOptions features; features.enabled = true;
        features.odometrySurfLeafSize = .001f;  // 先核对原始分类，再单独核对按ring体素滤波。
        const auto plane = Rays(0), edge = Rays(1);
        const auto flat = ExtractFeatures(plane, plane.cloud, features);
        const auto selected = ExtractFeatures(edge, edge.cloud, features);
        Require(flat.corner->empty() && flat.surface->size() > 1000, "flat plane classification");
        Require(selected.corner->size() >= 16 && selected.surface->size() > 1000, "LIO-SAM geometric edge not detected");
        const auto corners = Ids(selected.corner), surfaces = Ids(selected.surface);
        for (int id : corners) Require(!surfaces.count(id), "corner points leaked into raw surface set");

        std::vector<std::size_t> order(edge.cloud->size()); std::iota(order.begin(), order.end(), 0);
        std::mt19937 random(7); std::shuffle(order.begin(), order.end(), random);
        Scan shuffled; shuffled.cloud.reset(new PointCloudType);
        for (auto id : order) { shuffled.cloud->push_back(edge.cloud->points[id]); shuffled.rings.push_back(edge.rings[id]); }
        const auto reordered = ExtractFeatures(shuffled, shuffled.cloud, features);
        Require(Ids(reordered.corner) == corners && Ids(reordered.surface) == surfaces, "ring/column projection depends on input layout");
        auto strict = features; strict.edgeThreshold = 1e9f;
        Require(ExtractFeatures(edge, edge.cloud, strict).corner->empty(), "edgeThreshold has no effect");
        strict = features; strict.surfThreshold = 0;
        Require(!ExtractFeatures(plane, plane.cloud, strict).surface->empty(), "surface must retain unlabeled non-corner points like original LIO-SAM");
        strict = features; strict.odometrySurfLeafSize = .4f;
        const auto sampled = ExtractFeatures(plane, plane.cloud, strict);
        Require(!sampled.surface->empty() && sampled.surface->size() < flat.surface->size(), "surface voxel filter not applied");
        strict = features; strict.Horizon_SCAN = 512;
        Require(ExtractFeatures(plane, plane.cloud, strict).surface->size() < flat.surface->size(), "Horizon_SCAN does not control projection columns");
        strict = features; strict.N_SCAN = 8;
        const auto eight = ExtractFeatures(edge, edge.cloud, strict);
        for (int id : Ids(eight.corner)) Require(id < 8 * 241, "out-of-range rings accepted");
        for (int id : Ids(eight.surface)) Require(id < 8 * 241, "out-of-range rings leaked into surface");

        Scan echoes = edge; echoes.cloud.reset(new PointCloudType(*edge.cloud));
        Scan far_first; far_first.cloud.reset(new PointCloudType);
        for (std::size_t i = 0; i < edge.cloud->size(); ++i) {
            auto p = edge.cloud->points[i]; p.x *= 2; p.y *= 2; p.z *= 2; p.intensity += 10000;
            echoes.cloud->push_back(p); echoes.rings.push_back(edge.rings[i]);
            far_first.cloud->push_back(p); far_first.rings.push_back(edge.rings[i]);
        }
        for (std::size_t i = 0; i < edge.cloud->size(); ++i) {
            far_first.cloud->push_back(edge.cloud->points[i]); far_first.rings.push_back(edge.rings[i]);
        }
        Require(Ids(ExtractFeatures(echoes, echoes.cloud, features).corner) == corners, "projection did not retain first pixel return");
        const auto first_return = ExtractFeatures(far_first, far_first.cloud, features);
        Require(!first_return.corner->empty(), "far-first projection produced no test edges");
        for (int id : Ids(first_return.corner)) Require(id >= 10000, "projection silently replaced first return by near return");

        CloudPtr deskewed(new PointCloudType(*edge.cloud));
        for (auto& p : deskewed->points) { p.x += 1; p.y += 2; }
        const auto moved_selection = ExtractFeatures(edge, deskewed, features);
        Require(Ids(moved_selection.corner) == corners, "deskewing changed raw-depth feature labels");
        for (const auto& p : moved_selection.corner->points) {
            const auto& raw = edge.cloud->points[static_cast<std::size_t>(p.intensity)];
            Require(std::abs(p.x - raw.x - 1) < 1e-5 && std::abs(p.y - raw.y - 2) < 1e-5, "corner did not use deskewed coordinates");
        }
        auto ring_boundary = plane;
        ring_boundary.cloud.reset(new PointCloudType(*plane.cloud));
        for (std::size_t i = 0; i < ring_boundary.cloud->size(); ++i) if (ring_boundary.rings[i] % 2) {
            auto& p = ring_boundary.cloud->points[i]; p.x *= 2; p.y *= 2; p.z *= 2;
        }
        Require(ExtractFeatures(ring_boundary, ring_boundary.cloud, features).corner->empty(), "curvature crossed ring boundaries");
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

        // 高曲率候选足够多时，验证原LIO-SAM六分区、每区20个以及±5点抑制。
        Scan waves; waves.cloud.reset(new PointCloudType);
        for (int col = 0; col < 2000; ++col) {
            const double angle = (col - 1000) * 2 * std::acos(-1.0) / 2000;
            const double distance = 8 + (col % 2 ? .06 : -.06);
            PointType p; p.x = distance * std::cos(angle); p.y = distance * std::sin(angle); p.z = 0; p.intensity = col;
            waves.cloud->push_back(p); waves.rings.push_back(0);
        }
        auto many = features; many.N_SCAN = 1; many.edgeThreshold = .1f;
        const auto limited = ExtractFeatures(waves, waves.cloud, many);
        Require(limited.corner->size() == 120, "six sectors must select at most 20 corners each");
        const auto limited_ids = Ids(limited.corner);
        int previous = -100;
        for (int id : limited_ids) { Require(id - previous > 5, "corner neighborhood suppression missing"); previous = id; }

        NdtOptions options; options.num_layers = 2; options.num_threads = 1;
        options.resolutions = {.8, .4, .2}; options.weights = {1, 2, 3}; options.epsilon = 1e-6;
        const auto target = GeometryScene();
        MultiLayerNdt joint(options, features); joint.SetFeatureTarget(target);
        MultiLayerNdt corner(options, features), surface(options, features);
        corner.SetFeatureTarget({target.corner, nullptr}); surface.SetFeatureTarget({nullptr, target.surface});
        Vec6d pose; pose << .01, -.01, .005, .003, -.002, .004;
        Vec6d g, gc, gs; Mat6d h, hc, hs; std::vector<double> scores, unused;
        const double value = joint.EvaluateFeatures(target, pose, g, h, scores);
        const double c = corner.EvaluateFeatures({target.corner, nullptr}, pose, gc, hc, unused);
        const double s = surface.EvaluateFeatures({nullptr, target.surface}, pose, gs, hs, unused);
        Require(scores.size() == 4 && joint.VoxelCounts().size() == 4, "feature mode does not have two independent pyramids");
        Require(value > 0 && value <= 1 && std::abs(value - c - s) < 1e-10 && (g - gc - gs).norm() < 1e-8 &&
            (h - hc - hs).norm() < 1e-7, "geometric feature score/gradient/Hessian not jointly weighted");
        Require(std::abs(.5 * std::accumulate(scores.begin(), scores.end(), 0.0) - value) < 1e-10,
            "cross-layer geometric score was counted more than once");
        for (auto count : joint.VoxelCounts()) Require(count > 0, "occupied feature pyramid is empty");
        FeatureClouds swapped{target.surface, target.corner};
        Require(joint.EvaluateFeatures(swapped, Vec6d::Zero(), gc, hc, unused) == 0.0,
            "corner and surface can match the wrong target category");
        FeatureClouds surface_only{CloudPtr(new PointCloudType), target.surface};
        Require(std::abs(joint.EvaluateFeatures(surface_only, pose, gc, hc, unused) - s) < 1e-10,
            "missing corner points blocked valid surface observations");
        for (int axis = 0; axis < 6; ++axis) {
            Vec6d plus = pose, minus = pose, gp, gm; Mat6d hp, hm;
            const double step = 1e-5; plus(axis) += step; minus(axis) -= step;
            const double fp = joint.EvaluateFeatures(target, plus, gp, hp, unused);
            const double fm = joint.EvaluateFeatures(target, minus, gm, hm, unused);
            Require(std::abs((fp - fm) / (2 * step) - g(axis)) < 1e-4 * (1 + std::abs(g(axis))), "geometric feature gradient finite difference");
            Require(((gp - gm) / (2 * step) - h.col(axis)).norm() < 1e-4 * (1 + h.col(axis).norm()), "geometric feature Hessian finite difference");
        }
        auto parallel_options = options; parallel_options.num_threads = 4;
        MultiLayerNdt parallel(parallel_options, features); parallel.SetFeatureTarget(target);
        parallel.EvaluateFeatures(target, pose, gc, hc, unused);
        Require((gc - g).norm() < 1e-8 && (hc - h).norm() < 1e-6, "parallel geometry derivatives differ from serial");

        // 倾斜线/面覆盖非对角投影矩阵；仅轴对齐场景不能充分检验求导链。
        FeatureClouds tilted{CloudPtr(new PointCloudType), CloudPtr(new PointCloudType)};
        const Eigen::Matrix4f tilt = MultiLayerNdt::ToPose((Vec6d() << .3, -.4, .8, .27, -.31, .23).finished()).matrix().cast<float>();
        pcl::transformPointCloud(*target.corner, *tilted.corner, tilt);
        pcl::transformPointCloud(*target.surface, *tilted.surface, tilt);
        MultiLayerNdt oblique(options, features); oblique.SetFeatureTarget(tilted);
        const double oblique_score = oblique.EvaluateFeatures(tilted, pose, g, h, unused);
        const double score_only = oblique.EvaluateFeatures(tilted, pose, gc, hc, unused, false);
        Require(std::abs(score_only - oblique_score) < 1e-12 && (gc - g).norm() < 1e-10 && hc.norm() == 0,
            "score/gradient changed when Hessian calculation was disabled");
        for (int axis = 0; axis < 6; ++axis) {
            Vec6d plus = pose, minus = pose, gp, gm; Mat6d hp, hm;
            const double step = 1e-5; plus(axis) += step; minus(axis) -= step;
            const double fp = oblique.EvaluateFeatures(tilted, plus, gp, hp, unused);
            const double fm = oblique.EvaluateFeatures(tilted, minus, gm, hm, unused);
            Require(std::abs((fp - fm) / (2 * step) - g(axis)) < 1e-3 * (1 + std::abs(g(axis))), "oblique feature gradient finite difference");
            Require(((gp - gm) / (2 * step) - h.col(axis)).norm() < 1e-3 * (1 + h.col(axis).norm()), "oblique feature Hessian finite difference");
        }

        // 闭式分数检验几何投影、距离尺度和线/面的自由方向。
        CloudPtr line(new PointCloudType), plane_map(new PointCloudType), query(new PointCloudType);
        for (std::size_t i = 0; i < target.corner->size(); i += 3) line->push_back(target.corner->points[i]);
        for (std::size_t i = 0; i < target.surface->size(); i += 3) plane_map->push_back(target.surface->points[i]);
        PointType p = line->points[line->size() / 2]; p.y += .1f; p.z -= .08f; query->push_back(p);
        MultiLayerNdt single_line(options, features); single_line.SetFeatureTarget({line, nullptr});
        const double dy = p.y - line->points.front().y, dz = p.z - line->points.front().z;
        const double line_value = single_line.EvaluateFeatures({query, nullptr}, Vec6d::Zero(), gc, hc, unused);
        Require(std::abs(line_value - .5 * std::exp(-(dy * dy + dz * dz) / (2 * .3 * .3))) < 1e-8 && std::abs(gc(0)) < 1e-10,
            "corner score is not point-to-line squared negative exponential");
        Vec6d tangent = Vec6d::Zero(); tangent(0) = .12;
        Require(std::abs(single_line.EvaluateFeatures({query, nullptr}, tangent, gc, hc, unused) - line_value) < 1e-8,
            "line tangent direction was incorrectly constrained");
        auto wider = features; wider.corner_distance_scale = .6;
        MultiLayerNdt broad(options, wider); broad.SetFeatureTarget({line, nullptr});
        Require(broad.EvaluateFeatures({query, nullptr}, Vec6d::Zero(), gc, hc, unused) > line_value, "corner distance scale not applied");
        query->clear(); p = plane_map->points[plane_map->size() / 2]; p.z += .12f; query->push_back(p);
        MultiLayerNdt single_plane(options, features); single_plane.SetFeatureTarget({nullptr, plane_map});
        const double distance = p.z - plane_map->points.front().z;
        const double plane_value = single_plane.EvaluateFeatures({nullptr, query}, Vec6d::Zero(), gc, hc, unused);
        // 体素质心以float保存，累加舍入可使拟合平面产生微米级偏差。
        Require(std::abs(plane_value - .5 * std::exp(-distance * distance / (2 * .2 * .2))) < 1e-6 &&
            std::abs(gc(0)) < 1e-5 && std::abs(gc(1)) < 1e-5, "QR plane offset or projected plane distance incorrect");
        tangent(0) = .07; tangent(1) = -.08;
        Require(std::abs(single_plane.EvaluateFeatures({nullptr, query}, tangent, gc, hc, unused) - plane_value) < 1e-6,
            "plane tangent directions were incorrectly constrained");
        wider = features; wider.surface_distance_scale = .4;
        MultiLayerNdt broad_plane(options, wider); broad_plane.SetFeatureTarget({nullptr, plane_map});
        Require(broad_plane.EvaluateFeatures({nullptr, query}, Vec6d::Zero(), gc, hc, unused) > plane_value, "surface distance scale not applied");

        // 平面经过map原点时，也要得到同样的点到面距离和分数。
        for (auto& q : plane_map->points) q.z = 0.0f;
        query->points.front().z = static_cast<float>(distance);
        single_plane.SetFeatureTarget({nullptr, plane_map});
        Require(std::abs(single_plane.EvaluateFeatures({nullptr, query}, Vec6d::Zero(), gc, hc, unused) - plane_value) < 1e-6,
            "QR plane fit lost a plane through the map origin");
        auto small = options; small.resolutions = {.04, .02, .01}; small.num_layers = 3;
        CloudPtr few(new PointCloudType);
        for (int i = 0; i < 3; ++i) few->push_back(line->points[i]);
        MultiLayerNdt insufficient(small, features); insufficient.SetFeatureTarget({few, nullptr});
        Require(insufficient.EvaluateFeatures({few, nullptr}, Vec6d::Zero(), gc, hc, unused) == 0,
            "duplicate cross-layer centers manufactured sufficient support");
        MultiLayerNdt sparse(small, features); sparse.SetFeatureTarget({line, nullptr});
        for (auto count : sparse.VoxelCounts()) if (count) Require(count == line->size(), "sparse occupied feature voxels were discarded");
        MultiLayerNdt gaussian(small); gaussian.SetTarget(line);
        for (auto count : gaussian.VoxelCounts()) Require(count == 0, "full NDT no longer applies original six-point voxel requirement");
        // 精确共线的斜线：投影矩阵的舍入误差不能让接近零的距离变成零分。
        CloudPtr exact_line(new PointCloudType), exact_query(new PointCloudType);
        for (int i = -8; i <= 8; ++i) {
            const float t = .25f * i;
            PointType q; q.x = t; q.y = 2 * t + 1; q.z = 3 * t + 2; exact_line->push_back(q);
            q.x = t + .125f; q.y = 2 * q.x + 1; q.z = 3 * q.x + 2; exact_query->push_back(q);
        }
        auto wide_search = features; wide_search.neighbor_radius_factor = 1000;
        MultiLayerNdt exact(small, wide_search); exact.SetFeatureTarget({exact_line, nullptr});
        Require(std::abs(exact.EvaluateFeatures({exact_query, nullptr}, Vec6d::Zero(), gc, hc, unused) - .5) < 1e-8,
            "near-zero oblique line residual was discarded by exponential roundoff");
        CloudPtr cube(new PointCloudType);
        for (int i = 0; i < 8; ++i) {
            PointType q; q.x = 2.13f + (i & 1 ? .4f : -.4f); q.y = .17f + (i & 2 ? .4f : -.4f);
            q.z = 3.19f + (i & 4 ? .4f : -.4f); cube->push_back(q);
        }
        query->clear(); p.x = 2.14f; p.y = .18f; p.z = 3.2f; query->push_back(p);
        MultiLayerNdt invalid_geometry(options, features); invalid_geometry.SetFeatureTarget({cube, cube});
        Require(invalid_geometry.EvaluateFeatures({query, nullptr}, Vec6d::Zero(), gc, hc, unused) == 0,
            "PCA accepted non-linear corner neighborhood");
        Require(invalid_geometry.EvaluateFeatures({nullptr, query}, Vec6d::Zero(), gc, hc, unused) == 0,
            "QR accepted support farther than 0.2m from fitted plane");

        const SE3 truth = MultiLayerNdt::ToPose((Vec6d() << .04, -.025, .015, .004, -.003, .006).finished());
        FeatureClouds source{CloudPtr(new PointCloudType), CloudPtr(new PointCloudType)};
        const Eigen::Matrix4f inverse = truth.inverse().matrix().cast<float>();
        pcl::transformPointCloud(*target.corner, *source.corner, inverse);
        pcl::transformPointCloud(*target.surface, *source.surface, inverse);
        const auto fit = joint.AlignFeatures(source, SE3());
        const auto fit_error = truth.inverse() * fit.pose;
        Require(fit.valid && fit.corner_matches > 0 && fit.surface_matches > 0 &&
            fit_error.translation().norm() < .002 && fit_error.so3().log().norm() < .0002,
            "joint geometric feature pose optimization failed");
        const auto full_target = Clusters(0);
        MultiLayerNdt original_ndt(options); original_ndt.SetTarget(full_target);
        const double original_score = original_ndt.Evaluate(full_target, pose, gc, hc, unused);
        double expected = 0;
        for (int layer = 0; layer < 2; ++layer) {
            auto single = options; single.num_layers = 1;
            single.resolutions = {options.resolutions[layer]}; single.weights = {1};
            MultiLayerNdt one(single); one.SetTarget(full_target);
            expected += options.weights[layer] / 3.0 * one.Evaluate(full_target, pose, gc, hc, unused);
        }
        Require(std::abs(original_score - expected) < 1e-10, "num_layers does not select and normalize the first N levels");
        joint.SetTarget(full_target);
        Require(joint.VoxelCounts().size() == 2 && std::abs(joint.Evaluate(full_target, pose, gc, hc, unused) - original_score) < 1e-10,
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
