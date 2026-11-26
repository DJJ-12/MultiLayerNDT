#pragma once

#include "common/struct.h"
#include "core/mapping/multi_layer_ndt.h"

namespace lightning {
class Mapping {
public:
    explicit Mapping(MappingOptions options);
    MappingResult Process(const Scan& scan);
    const MapData& Data() const { return data_; }
    static bool SaveMap(const MapData& data, const std::string& directory, double voxel_size);

private:
    CloudPtr Downsample(const CloudPtr& cloud, double size) const;
    bool UpdateSubmap(const SE3& prediction);
    MappingOptions options_;
    MultiLayerNdt ndt_;
    MapData data_;
    std::vector<std::size_t> active_;
    CloudPtr submap_;
};
}  // namespace lightning
