#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_msgs/msg/header.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include "core/mapping/mapping.h"
#ifdef MULTI_LAYER_NDT_WITH_UI
#include "ui/pangolin_window.h"
#endif

namespace lightning {
class MappingNode : public rclcpp::Node {
public:
    MappingNode(const std::string& yaml_path, const std::string& save_path, bool no_ui);
    ~MappingNode();
    void Subscribe();
    void ProcessCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud);
    void Finish();
    bool Save();
    bool Alive();
    bool HasUI() const;
    const std::string& LidarTopic() const { return lidar_topic_; }

private:
    Scan Convert(const sensor_msgs::msg::PointCloud2& message) const;
    void Publish(const MappingResult& result, const std_msgs::msg::Header& header);
    std::unique_ptr<Mapping> mapping_;
    std::mutex processing_mutex_, saving_mutex_;
    std::string lidar_topic_, time_field_, map_frame_, lidar_frame_, save_path_;
    double time_scale_ = 1.0, min_range_ = 1.0, max_range_ = 100.0, save_voxel_ = .1;
    bool deskew_ = true, pub_tf_ = true, time_absolute_ = false;
    std::size_t frames_ = 0;
    nav_msgs::msg::Path path_;
    std_msgs::msg::Header last_header_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr scan_pub_, map_pub_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_;
    rclcpp::CallbackGroup::SharedPtr service_group_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_service_;
#ifdef MULTI_LAYER_NDT_WITH_UI
    std::shared_ptr<ui::PangolinWindow> ui_;
#endif
};
}  // namespace lightning
