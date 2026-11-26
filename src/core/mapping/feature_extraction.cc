#include "core/mapping/feature_extraction.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lightning {
namespace {
void ExtractSegment(const std::vector<std::size_t>& ids, std::size_t begin, std::size_t end,
                    const std::vector<double>& ranges, const CloudPtr& deskewed,
                    const FeatureOptions& options, FeatureClouds& features) {
    const int n = static_cast<int>(end - begin);
    const int high = options.highpass_half_window, low = options.lowpass_half_window;
    const int border = std::max(high, low);
    std::vector<double> depth(n), sum(n + 1, 0.0), energy(n + 1, 0.0);
    std::vector<bool> usable(n, true), corner(n, false);
    for (int i = 0; i < n; ++i) {
        depth[i] = ranges[ids[begin + i]];
        sum[i + 1] = sum[i] + depth[i];
    }
    for (int i = 0; i + 1 < n; ++i) {
        const double jump = options.occlusion_depth_ratio * std::min(depth[i], depth[i + 1]);
        // 遮挡跳变只屏蔽远侧，近侧边缘仍可作为corner候选。
        if (depth[i] - depth[i + 1] > jump)
            for (int j = std::max(0, i - border + 1); j <= i; ++j) usable[j] = false;
        else if (depth[i + 1] - depth[i] > jump)
            for (int j = i + 1; j <= std::min(n - 1, i + border); ++j) usable[j] = false;
    }
    for (int i = 1; i + 1 < n; ++i) {
        const double left = depth[i] - depth[i - 1], right = depth[i] - depth[i + 1];
        const double threshold = options.occlusion_depth_ratio * depth[i];
        if (left * right > 0 && std::abs(left) > threshold && std::abs(right) > threshold) usable[i] = false;
    }
    // 保留响应带内的点，不按每段固定个数限额，给NDT高斯体素保留足够支持点。
    for (int i = high; i + high < n; ++i) {
        const double smooth = (sum[i + high + 1] - sum[i - high]) / (2 * high + 1);
        const double response = std::abs(depth[i] - smooth) / depth[i];
        if (usable[i] && response > options.highpass_threshold) {
            corner[i] = true;
            features.corner->push_back(deskewed->points[ids[begin + i]]);
        }
    }
    // 低通后的残差能量，比仅检查中心点残差更能识别正负波动互相抵消的噪声。
    for (int i = 0; i < n; ++i) {
        double residual = 0.0;
        if (i >= low && i + low < n)
            residual = depth[i] - (sum[i + low + 1] - sum[i - low]) / (2 * low + 1);
        energy[i + 1] = energy[i] + residual * residual;
    }
    for (int i = 2 * low; i + 2 * low < n; ++i) {
        const double mean_energy = (energy[i + low + 1] - energy[i - low]) / (2 * low + 1);
        const double roughness = std::sqrt(std::max(0.0, mean_energy)) / depth[i];
        if (usable[i] && !corner[i] && roughness < options.lowpass_threshold)
            features.surface->push_back(deskewed->points[ids[begin + i]]);
    }
}
}  // namespace

FeatureClouds ExtractFeatures(const Scan& scan, const CloudPtr& deskewed, const FeatureOptions& options) {
    if (options.highpass_half_window < 1 || options.lowpass_half_window < 1)
        throw std::invalid_argument("特征滤波的half_window必须至少为1");
    if (scan.rings.size() != scan.cloud->size() || deskewed->size() != scan.cloud->size())
        throw std::invalid_argument("特征模式需要与有效点一一对应的ring字段");
    FeatureClouds features{CloudPtr(new PointCloudType), CloudPtr(new PointCloudType)};
    std::map<std::uint16_t, std::vector<std::size_t>> rings;
    std::vector<double> angles(scan.cloud->size()), ranges(scan.cloud->size());
    for (std::size_t i = 0; i < scan.cloud->size(); ++i) {
        const auto& point = scan.cloud->points[i];
        angles[i] = std::atan2(point.y, point.x);
        ranges[i] = point.getVector3fMap().cast<double>().norm();
        if (ranges[i] > 0.0 && std::isfinite(ranges[i])) rings[scan.rings[i]].push_back(i);
    }
    const double max_gap = options.max_gap_deg * std::acos(-1.0) / 180.0;
    for (auto& entry : rings) {
        auto& ids = entry.second;
        std::sort(ids.begin(), ids.end(), [&](auto a, auto b) {
            return angles[a] == angles[b] ? ranges[a] < ranges[b] : angles[a] < angles[b];
        });
        // 同一方向多回波保留近点，避免零角间隔的远近回波被当成相邻结构。
        std::size_t count = 0;
        for (auto id : ids) {
            if (count > 0 && std::abs(angles[id] - angles[ids[count - 1]]) < 1e-6) {
                if (ranges[id] < ranges[ids[count - 1]]) ids[count - 1] = id;
            } else ids[count++] = id;
        }
        ids.resize(count);
        std::size_t begin = 0;
        for (std::size_t end = 1; end <= ids.size(); ++end) {
            if (end < ids.size() && angles[ids[end]] - angles[ids[end - 1]] <= max_gap) continue;
            ExtractSegment(ids, begin, end, ranges, deskewed, options, features);
            begin = end;
        }
    }
    return features;
}
}  // namespace lightning
