#pragma once

#include <functional>
#include <string>
#include <utility>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace lightning {
// 离线直接调用同一接收回调，不经过DDS，不跳帧。
class RosbagIO {
public:
    using Handler = std::function<bool(sensor_msgs::msg::PointCloud2::SharedPtr)>;
    explicit RosbagIO(std::string bag_file) : bag_file_(std::move(bag_file)) {}
    RosbagIO& AddPointCloud2Handle(std::string topic, Handler handler) {
        topic_ = std::move(topic); handler_ = std::move(handler); return *this;
    }
    void Go();
private:
    std::string bag_file_, topic_;
    Handler handler_;
};
}  // namespace lightning
