#pragma once

#include <msg_out/msg/jzxyimu.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <stdexcept>

#include "core/lightning_math.hpp"
#include "common/struct.h"

namespace lightning {

// 建图、定位共用：整理厂家GPS数据，把第一帧有效GPS对齐到当时最新一帧雷达的时间，下层只用timestamp。
inline GPS ConvertGps(const msg_out::msg::JZXYINS::SharedPtr& gps, double latest_lidar_stamp,
                      double& first_gps_stamp, double& gps_lidar_stamp, bool relative) {
    GPS data;
    data.raw = gps;
    data.device_timestamp = math::GpsWeekToSeconds(gps->weekno, gps->secondofweek);
    if (!relative) data.timestamp = data.device_timestamp;
    else {
        // 锚点取第一帧有效GPS到达时最新一帧雷达的时间戳，而不是程序启动后的首帧雷达。
        if (first_gps_stamp < 0.0 && gps->alignstatus == 3 && latest_lidar_stamp >= 0.0) {
            first_gps_stamp = data.device_timestamp;
            gps_lidar_stamp = latest_lidar_stamp;
        }
        if (first_gps_stamp >= 0.0) data.timestamp = gps_lidar_stamp + (data.device_timestamp - first_gps_stamp);
    }
    data.lla = Vec3d(gps->latitude, gps->longitude, gps->altitude + gps->ellipsoidalalt);
    data.rpy_ned = Vec3d(gps->roll, gps->pitch, gps->courseang) * math::kDegreesToRadians;
    data.velocity_enu = Vec3d(gps->eastspeed, gps->northspeed, gps->upperspeed);
    data.position_covariance_enu = Vec3d(gps->logstd, gps->latstd, gps->altstd).array().square().matrix().asDiagonal();
    data.velocity_covariance_enu = Vec3d(gps->eastspeedstd, gps->northspeedstd, gps->upperspeedstd).array().square().matrix().asDiagonal();
    data.attitude_covariance_ned = (Vec3d(gps->rollstd, gps->pitchstd, gps->courseangstd) * math::kDegreesToRadians).array().square().matrix().asDiagonal();
    data.alignstatus = gps->alignstatus;
    data.budrstatus = gps->budrstatus;
    return data;
}

// 在线接收和离线读包共用：对齐雷达时间，将厂家IMU转换为建图使用的标准消息。
inline sensor_msgs::msg::Imu::SharedPtr ConvertImu(const msg_out::msg::JZXYIMU& msg,
                                                 double first_lidar_stamp, double& first_imu_stamp) {
    const double stamp = math::GpsWeekToSeconds(msg.weekno, msg.secondofweek);
    if (first_imu_stamp < 0.0) first_imu_stamp = stamp;
    if (first_lidar_stamp < 0.0) return nullptr;

    auto imu = std::make_shared<sensor_msgs::msg::Imu>();
    imu->header.stamp = math::FromSec(first_lidar_stamp + (stamp - first_imu_stamp));
    imu->header.frame_id = "imu_link";
    imu->angular_velocity.x = msg.gyro_x * math::kDegreesToRadians;
    imu->angular_velocity.y = msg.gyro_y * math::kDegreesToRadians;
    imu->angular_velocity.z = msg.gyro_z * math::kDegreesToRadians;
    imu->linear_acceleration.x = msg.acce_x;
    imu->linear_acceleration.y = msg.acce_y;
    imu->linear_acceleration.z = msg.acce_z;
    // 厂家IMU没有姿态；单位四元数只作占位，姿态初始化沿用建图中的加速度处理。
    imu->orientation.w = 1.0;
    imu->orientation_covariance[0] = -1.0;
    return imu;
}

}  // namespace lightning
