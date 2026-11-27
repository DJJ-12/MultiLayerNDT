#include "core/mapping/mapping.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <set>

#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/approximate_progressive_morphological_filter.h>

namespace lightning {
namespace {
namespace fs = std::filesystem;

fs::path PendingFile(fs::path path) {
    path += ".tmp";
    return path;
}

bool SaveWithoutGround(const fs::path& path, const CloudPtr& map) {
    // 用XY网格上的局部地面高度处理坡路。距离和网格边长单位为米。
    pcl::ApproximateProgressiveMorphologicalFilter<PointType> terrain;
    terrain.setInputCloud(map);
    terrain.setCellSize(1.0f);
    terrain.setMaxWindowSize(9);
    terrain.setSlope(0.1f);
    terrain.setInitialDistance(0.18f);
    terrain.setMaxDistance(0.45f);
    terrain.setNumberOfThreads(1);
    pcl::PointIndices::Ptr ground(new pcl::PointIndices);
    terrain.extract(ground->indices);

    pcl::ExtractIndices<PointType> keep;
    keep.setInputCloud(map);
    keep.setIndices(ground);
    keep.setNegative(true);
    PointCloudType objects;
    keep.filter(objects);
    if (!objects.empty()) return pcl::io::savePCDFileBinaryCompressed(path.string(), objects) == 0;

    // PCL 1.12的PCD写入接口拒绝空点云，直接写同字段的零点头。
    std::ofstream empty(path, std::ios::binary);
    empty << pcl::PCDWriter::generateHeader(objects, 0) << "DATA binary\n";
    empty.close();
    return static_cast<bool>(empty);
}

bool SaveBlockMap(const fs::path& directory, const PointCloudType& map, int resolution, double voxel_size) {
    using BlockCloud = pcl::PointCloud<pcl::PointXYZ>;
    std::map<std::pair<int, int>, BlockCloud::Ptr> blocks;
    for (const auto& point : map.points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) continue;
        // 与lightning-lm一致，索引为块原点的米坐标，负坐标向下取整。
        const int x = static_cast<int>(std::floor(point.x / double(resolution))) * resolution;
        const int y = static_cast<int>(std::floor(point.y / double(resolution))) * resolution;
        auto& block = blocks[{x, y}];
        if (!block) block.reset(new BlockCloud);
        pcl::PointXYZ xyz;
        xyz.x = point.x; xyz.y = point.y; xyz.z = point.z;
        block->push_back(xyz);
    }
    if (blocks.empty()) return false;

    const auto block_dir = directory / "BlockMap";
    const auto pcd_dir = block_dir / "pointcloud_map";
    const auto metadata_path = block_dir / "pointcloud_map_metadata.yaml";
    fs::create_directories(pcd_dir);

    // 重存时只清理上一次索引中失效的块文件。
    std::set<fs::path> stale;
    {
        std::ifstream previous(metadata_path);
        for (std::string line; std::getline(previous, line);) {
            const fs::path name = line.substr(0, line.find(':'));
            if (line.compare(0, 4, "seg_") == 0 && name == name.filename() && name.extension() == ".pcd")
                stale.insert(name);
        }
    }
    std::ofstream metadata(PendingFile(metadata_path));
    if (!metadata) return false;
    metadata << "x_resolution: " << resolution << "\ny_resolution: " << resolution << '\n';

    std::vector<fs::path> files;
    for (const auto& [origin, block] : blocks) {
        BlockCloud sampled;
        const BlockCloud* output = block.get();
        if (voxel_size > 0) {
            pcl::VoxelGrid<pcl::PointXYZ> filter;
            filter.setLeafSize(voxel_size, voxel_size, voxel_size);
            filter.setInputCloud(block);
            filter.filter(sampled);
            output = &sampled;
        }
        if (output->empty()) continue;

        const std::string filename = "seg_" + std::to_string(origin.first) + "_" + std::to_string(origin.second) + ".pcd";
        const auto path = pcd_dir / filename;
        if (pcl::io::savePCDFileBinary(PendingFile(path).string(), *output) != 0) return false;
        metadata << filename << ": [" << origin.first << ", " << origin.second << "]\n";
        files.push_back(path);
        stale.erase(filename);
    }
    metadata.close();
    if (files.empty() || !metadata) return false;

    for (const auto& path : files) fs::rename(PendingFile(path), path);
    fs::rename(PendingFile(metadata_path), metadata_path);
    for (const auto& name : stale) fs::remove(pcd_dir / name);
    return true;
}

void WritePose(std::ostream& file, double stamp, const SE3& pose) {
    const auto& t = pose.translation();
    const auto q = pose.unit_quaternion();
    file << stamp << ' ' << t.x() << ' ' << t.y() << ' ' << t.z() << ' '
         << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
}
}  // namespace

bool Mapping::SaveMap(const MapData& data, const std::string& directory, double voxel_size,
                      int block_resolution, double block_voxel_size) {
    if (data.keyframes.empty() || directory.empty() || !(voxel_size > 0) || block_resolution <= 0) return false;
    CloudPtr map(new PointCloudType), sampled(new PointCloudType);
    for (const auto& frame : data.keyframes) {
        PointCloudType world;
        const Eigen::Matrix4f transform = frame.pose.matrix().cast<float>();
        pcl::transformPointCloud(*frame.cloud, world, transform);
        *map += world;
    }
    if (map->empty()) return false;
    {
        pcl::VoxelGrid<PointType> filter;
        filter.setLeafSize(voxel_size, voxel_size, voxel_size);
        filter.setInputCloud(map);
        filter.filter(*sampled);
    }
    map.reset();
    if (sampled->empty()) return false;

    const fs::path root(directory);
    fs::create_directories(root);
    const auto global_path = root / "global.pcd";
    if (pcl::io::savePCDFileBinaryCompressed(PendingFile(global_path).string(), *sampled) != 0) return false;
    const auto display_path = root / "global_no_ground.pcd";
    if (!SaveWithoutGround(PendingFile(display_path), sampled)) return false;

    const auto trajectory_path = root / "trajectory.txt";
    std::ofstream trajectory(PendingFile(trajectory_path));
    trajectory << std::fixed << std::setprecision(9);
    for (const auto& state : CorrectedTrajectory(data)) WritePose(trajectory, state.timestamp_, state.pose);
    trajectory.close();

    const auto poses_path = root / "pose.txt";
    std::ofstream poses(PendingFile(poses_path));
    poses << "# parent_frame: lidar0\n# child_frame: lidar\n# pose: T_L0_Lk\n"
          << "# id timestamp tx ty tz qx qy qz qw\n" << std::setprecision(18);
    for (std::size_t id = 0; id < data.keyframes.size(); ++id) {
        const auto& frame = data.keyframes[id];
        poses << id << ' ';
        WritePose(poses, frame.stamp, frame.pose);
    }
    poses.close();

    if (!trajectory || !poses || !SaveBlockMap(root, *sampled, block_resolution, block_voxel_size)) return false;
    fs::rename(PendingFile(global_path), global_path);
    fs::rename(PendingFile(display_path), display_path);
    fs::rename(PendingFile(trajectory_path), trajectory_path);
    fs::rename(PendingFile(poses_path), poses_path);
    return true;
}
}  // namespace lightning
