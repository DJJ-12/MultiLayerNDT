#include "core/mapping/feature_extraction.h"
#include <pcl/filters/voxel_grid.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace lightning {
namespace {
// 移植自原LIO-SAM featureExtraction.cpp：±5点曲率、遮挡检查、六分区选点与邻域抑制。
void ExtractRing(const std::vector<int>& ids, const std::vector<int>& columns, const Scan& scan,
                 const CloudPtr& deskewed, const FeatureOptions& options, FeatureClouds& features) {
    const int n = static_cast<int>(ids.size());
    if (n < 11) return;
    std::vector<float> range(n), curvature(n, 0.0f);
    std::vector<int> order(n), label(n, 0);
    std::vector<bool> picked(n, false);
    std::iota(order.begin(), order.end(), 0);
    for (int i = 0; i < n; ++i) range[i] = scan.cloud->points[ids[i]].getVector3fMap().norm();
    // 前后各五点的窗口严格留在本ring中，不借用相邻ring末端的数据。
    for (int i = 5; i < n - 5; ++i) {
        const float diffRange = range[i-5] + range[i-4] + range[i-3] + range[i-2] + range[i-1]
            - range[i] * 10 + range[i+1] + range[i+2] + range[i+3] + range[i+4] + range[i+5];
        curvature[i] = diffRange * diffRange;
    }
    for (int i = 5; i < n - 6; ++i) {
        if (std::abs(columns[i + 1] - columns[i]) < 10) {
            if (range[i] - range[i + 1] > 0.3f)
                for (int j = i - 5; j <= i; ++j) picked[j] = true;
            else if (range[i + 1] - range[i] > 0.3f)
                for (int j = i + 1; j <= i + 6; ++j) picked[j] = true;
        }
        const float diff1 = std::abs(range[i - 1] - range[i]);
        const float diff2 = std::abs(range[i + 1] - range[i]);
        if (diff1 > 0.02f * range[i] && diff2 > 0.02f * range[i]) picked[i] = true;
    }
    const auto suppressNeighbors = [&](int index) {
        picked[index] = true;
        for (int step : {-1, 1}) for (int offset = 1; offset <= 5; ++offset) {
            const int current = index + step * offset, previous = current - step;
            if (current < 0 || current >= n || std::abs(columns[current] - columns[previous]) > 10) break;
            picked[current] = true;
        }
    };
    CloudPtr surface(new PointCloudType);
    const int valid = n - 10;
    for (int section = 0; section < 6; ++section) {
        const int sp = 5 + valid * section / 6;
        const int ep = 5 + valid * (section + 1) / 6 - 1;
        if (sp >= ep) continue;
        // ep为闭区间末端，排序包含ep，避免末尾候选漏排。
        std::sort(order.begin() + sp, order.begin() + ep + 1, [&](int a, int b) {
            return curvature[a] == curvature[b] ? a < b : curvature[a] < curvature[b];
        });
        int largestPickedNum = 0;
        for (int k = ep; k >= sp; --k) {
            const int index = order[k];
            if (!picked[index] && curvature[index] > options.edgeThreshold) {
                if (++largestPickedNum > 20) break;
                label[index] = 1;
                features.corner->push_back(deskewed->points[ids[index]]);
                suppressNeighbors(index);
            }
        }
        for (int k = sp; k <= ep; ++k) {
            const int index = order[k];
            if (!picked[index] && curvature[index] < options.surfThreshold) {
                label[index] = -1;
                suppressNeighbors(index);
            }
        }
        // 沿用原LIO-SAM：surface包含非corner点，而非只保留label=-1的少量平坦点。
        for (int k = sp; k <= ep; ++k) if (label[k] <= 0) surface->push_back(deskewed->points[ids[k]]);
    }
    if (!surface->empty()) {
        PointCloudType sampled;
        pcl::VoxelGrid<PointType> filter;
        filter.setLeafSize(options.odometrySurfLeafSize, options.odometrySurfLeafSize, options.odometrySurfLeafSize);
        filter.setInputCloud(surface); filter.filter(sampled);
        *features.surface += sampled;
    }
}
}  // namespace

FeatureClouds ExtractFeatures(const Scan& scan, const CloudPtr& deskewed, const FeatureOptions& options) {
    if (options.N_SCAN < 1 || options.Horizon_SCAN < 1 || options.odometrySurfLeafSize <= 0)
        throw std::invalid_argument("LIO-SAM特征参数N_SCAN/Horizon_SCAN/odometrySurfLeafSize必须为正");
    if (scan.rings.size() != scan.cloud->size() || deskewed->size() != scan.cloud->size())
        throw std::invalid_argument("特征模式需要与有效点一一对应的ring字段");
    FeatureClouds features{CloudPtr(new PointCloudType), CloudPtr(new PointCloudType)};
    std::vector<int> image(static_cast<std::size_t>(options.N_SCAN) * options.Horizon_SCAN, -1);
    const float angular_resolution = 360.0f / options.Horizon_SCAN;
    for (std::size_t i = 0; i < scan.cloud->size(); ++i) {
        const int row = scan.rings[i];
        if (row >= options.N_SCAN) continue;
        const auto& point = scan.cloud->points[i];
        const float horizonAngle = std::atan2(point.x, point.y) * 180.0f / static_cast<float>(std::acos(-1.0));
        int column = -static_cast<int>(std::round((horizonAngle - 90.0f) / angular_resolution)) + options.Horizon_SCAN / 2;
        if (column >= options.Horizon_SCAN) column -= options.Horizon_SCAN;
        if (column < 0 || column >= options.Horizon_SCAN) continue;
        auto& cell = image[static_cast<std::size_t>(row) * options.Horizon_SCAN + column];
        if (cell < 0) cell = static_cast<int>(i);  // 同一格保留第一点，沿用原projectPointCloud规则。
    }
    for (int ring = 0; ring < options.N_SCAN; ++ring) {
        std::vector<int> ids, columns;
        for (int column = 0; column < options.Horizon_SCAN; ++column) {
            const int id = image[static_cast<std::size_t>(ring) * options.Horizon_SCAN + column];
            if (id >= 0) { ids.push_back(id); columns.push_back(column); }
        }
        ExtractRing(ids, columns, scan, deskewed, options, features);
    }
    return features;
}
}  // namespace lightning
