//
// Created by xiang on 25-3-24.
//

#ifndef LIGHTNING_ROS_UTILS_H
#define LIGHTNING_ROS_UTILS_H

#include <atomic>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>

#include "common/point_def.h"

namespace lightning {

namespace debug {
inline std::atomic_bool flg_exit{false};  // 原版RosbagIO使用的退出标志。
}

inline double ToSec(const builtin_interfaces::msg::Time &time) { return double(time.sec) + 1e-9 * time.nanosec; }
inline uint64_t ToNanoSec(const builtin_interfaces::msg::Time &time) { return time.sec * 1e9 + time.nanosec; }

}  // namespace lightning

#endif  // LIGHTNING_ROS_UTILS_H
