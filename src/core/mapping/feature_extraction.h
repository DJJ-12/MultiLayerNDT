#pragma once
#include "common/struct.h"

namespace lightning {
// 原LIO-SAM的ring/列投影与特征选择；输入原始扫描，输出去畸变corner和按ring降采样的surface。
FeatureClouds ExtractFeatures(const Scan& scan, const CloudPtr& deskewed, const FeatureOptions& options);
}  // namespace lightning
