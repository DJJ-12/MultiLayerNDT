#include "wrapper/bag_io.h"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_filter.hpp>
#include <stdexcept>

namespace lightning {
void RosbagIO::Go() {
    rosbag2_cpp::Reader reader;
    reader.open({bag_file_, "sqlite3"}, {"cdr", "cdr"});
    bool found = false;
    for (const auto& topic : reader.get_all_topics_and_types()) {
        if (topic.name != topic_) continue;
        if (topic.type != "sensor_msgs/msg/PointCloud2") throw std::runtime_error("配置的雷达topic不是PointCloud2");
        found = true;
    }
    if (!found) throw std::runtime_error("bag中找不到雷达topic: " + topic_);
    rosbag2_storage::StorageFilter filter;
    filter.topics = {topic_}; reader.set_filter(filter);
    rclcpp::Serialization<sensor_msgs::msg::PointCloud2> serialization;
    while (rclcpp::ok() && reader.has_next()) {
        const auto raw = reader.read_next();
        auto cloud = std::make_shared<sensor_msgs::msg::PointCloud2>();
        rclcpp::SerializedMessage data(*raw->serialized_data);
        serialization.deserialize_message(&data, cloud.get());
        if (!handler_(cloud)) break;
    }
}
}  // namespace lightning
