#pragma once

#include "common/struct.h"
#include "core/mapping/multi_layer_ndt.h"
#include "core/backend/pose_graph.h"

namespace lightning {
class Mapping {
public:
    explicit Mapping(MappingOptions options);
    MappingResult Process(const Scan& scan);
    MappingResult Finish();  // 等最后一次回环完成并应用，读包结束后再保存。
    const MapData& Data() const { return data_; }
    static std::vector<NavState> CorrectedTrajectory(const MapData& data);
    static bool SaveMap(const MapData& data, const std::string& directory, double voxel_size,
                        int block_resolution = 20, double block_voxel_size = 0.1);

private:
    CloudPtr Downsample(const CloudPtr& cloud, double size) const;
    bool UpdateSubmap(const SE3& prediction);
    bool UpdateGraph();
    MappingOptions options_;
    MultiLayerNdt ndt_;
    PoseGraph backend_;
    MapData data_;
    SE3 map_from_odom_;
    std::vector<std::size_t> active_;
    std::map<int, CloudPtr> active_clouds_;
    CloudPtr submap_, last_scan_;
};
}  // namespace lightning
