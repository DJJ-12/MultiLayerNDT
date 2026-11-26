#pragma once
#include "common/struct.h"

namespace lightning {
// 用原扫描的ring/方位角/深度选点，输出对应的去畸变坐标，不移动或平滑点的位置。
FeatureClouds ExtractFeatures(const Scan& scan, const CloudPtr& deskewed, const FeatureOptions& options);
}  // namespace lightning
