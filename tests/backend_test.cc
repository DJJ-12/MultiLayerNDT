#include "core/mapping/mapping.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace lightning;
void Require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
SE3 Translation(double x, double y = 0.0) { return SE3(SO3(), Vec3d(x, y, 0.0)); }

int main() {
    try {
        LoopOptions options; options.loopClosureEnableFlag = false;
        PoseGraph graph(options);
        MapData data;
        CloudPtr cloud(new PointCloudType);
        PointType point; point.x = 1; point.y = 2; point.z = 3; point.intensity = 1; cloud->push_back(point);
        // 正方形走回起点，连续里程计末端累计0.8m漂移。
        for (int i = 0; i <= 40; ++i) {
            double x = 0, y = 0;
            if (i <= 10) x = i;
            else if (i <= 20) { x = 10; y = i - 10; }
            else if (i <= 30) { x = 30 - i; y = 10; }
            else y = 40 - i;
            const SE3 odom = Translation(x + .02 * i, y);
            data.keyframes.push_back({double(i), odom, cloud, odom});
            NavState state; state.timestamp_ = i; state.pose = odom; data.trajectory.push_back(state);
            graph.Update(data.keyframes);
        }
        const double before = data.keyframes.back().pose.translation().norm();
        graph.QueueLoop({40, 0, SE3(), 1e-6});
        Require(graph.Update(data.keyframes) && graph.LoopCount() == 1, "loop factor was not applied");
        const double after = data.keyframes.back().pose.translation().norm();
        Require(before > .79 && after < .05, "pose graph did not close drifted square");
        Require(data.keyframes.front().pose.translation().norm() < 1e-5, "first-frame origin moved");
        Require((data.keyframes.back().odom_pose.translation() - Vec3d(.8, 0, 0)).norm() < 1e-10 &&
            (data.trajectory.back().pose.translation() - Vec3d(.8, 0, 0)).norm() < 1e-10,
            "graph correction leaked into frontend motion history");
        const auto corrected = Mapping::CorrectedTrajectory(data);
        Require((corrected.back().pose.translation() - data.keyframes.back().pose.translation()).norm() < 1e-8,
            "exported trajectory ignored graph estimate");
        Require(cloud->front().x == 1 && cloud->front().y == 2, "backend rewrote local keyframe points");

        // 相邻关键帧之间的逐帧轨迹平滑应用图校正，两个端点严格对齐关键帧。
        MapData segment;
        segment.keyframes = {{0, SE3(), cloud, SE3()}, {2, Translation(1), cloud, Translation(2)}};
        for (int i = 0; i <= 2; ++i) { NavState state; state.timestamp_ = i; state.pose = Translation(i); segment.trajectory.push_back(state); }
        const auto middle = Mapping::CorrectedTrajectory(segment);
        Require(std::abs(middle[1].pose.translation().x() - .5) < 1e-10, "trajectory correction interpolation");
        const auto destination = std::filesystem::temp_directory_path() /
            ("ndt_backend_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        Require(Mapping::SaveMap(data, destination.string(), .1), "optimized map save failed");
        PointCloudType saved;
        Require(pcl::io::loadPCDFile<PointType>((destination / "map.pcd").string(), saved) == 0 && saved.size() > 20,
            "map export did not transform local keyframes by optimized poses");
        std::ifstream file(destination / "trajectory.txt");
        double stamp, x, y, z, qx, qy, qz, qw;
        Vec3d last_exported = Vec3d::Zero();
        while (file >> stamp >> x >> y >> z >> qx >> qy >> qz >> qw) last_exported = Vec3d(x, y, z);
        Require((last_exported - data.keyframes.back().pose.translation()).norm() < 1e-7, "saved trajectory contains pre-loop drift");

        // 距离/时间候选、真实ICP验证、异步提交和末尾等待整条回环链路。
        CloudPtr scene(new PointCloudType);
        std::mt19937 random(19);
        std::uniform_real_distribution<float> spread(-8, 8);
        for (int i = 0; i < 12000; ++i) {
            PointType p; p.x = spread(random); p.y = spread(random); p.z = .4f * spread(random); p.intensity = 1;
            scene->push_back(p);
        }
        options.loopClosureEnableFlag = true; options.mappingSurfLeafSize = .2;
        PoseGraph asynchronous(options);
        std::vector<Keyframe> revisit{{1, SE3(), scene, SE3()}, {41, Translation(.15), scene, Translation(.15)}};
        asynchronous.Update(revisit);
        asynchronous.CheckLoop(revisit, true); asynchronous.WaitForLoop();
        Require(asynchronous.Update(revisit) && asynchronous.LoopCount() == 1, "asynchronous loop/ICP chain failed");
        // 体素化残差决定原LIO-SAM的回环权重，单条里程计边不一定被完全校正。
        Require(revisit.back().pose.translation().norm() < .15, "ICP loop correction increased drift");
        asynchronous.CheckLoop(revisit, true); asynchronous.WaitForLoop();
        Require(!asynchronous.Update(revisit) && asynchronous.LoopCount() == 1, "duplicate loop added twice");

        options.historyKeyframeSearchTimeDiff = 100;
        PoseGraph recent(options);
        std::vector<Keyframe> nearby = revisit;
        recent.Update(nearby); recent.CheckLoop(nearby, true); recent.WaitForLoop();
        Require(!recent.Update(nearby) && recent.LoopCount() == 0, "recent scans produced a false loop");

        options.historyKeyframeSearchTimeDiff = 30;
        PoseGraph self_match(options);
        CloudPtr unrelated(new PointCloudType(*scene));
        for (auto& p : unrelated->points) p.x += 50;
        std::vector<Keyframe> false_revisit{{1, SE3(), scene, SE3()}, {41, Translation(.15), unrelated, Translation(.15)}};
        self_match.Update(false_revisit); self_match.CheckLoop(false_revisit, true); self_match.WaitForLoop();
        Require(!self_match.Update(false_revisit) && self_match.LoopCount() == 0, "ICP target contains current-frame points");
        std::cout << "Backend tests passed: closure " << before << " -> " << after
                  << " m; motion isolation, trajectory/export and asynchronous ICP chain\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
