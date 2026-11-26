#include <ament_index_cpp/get_package_share_directory.hpp>
#include <glog/logging.h>
#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
#include "wrapper/bag_io.h"
#include "wrapper/mapping_node.h"

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    rclcpp::init(argc, argv);
    const auto arguments = rclcpp::remove_ros_arguments(argc, argv);
    std::string config, bag, output;
    bool no_ui = false;
    try {
        for (std::size_t i = 1; i < arguments.size(); ++i) {
            const auto& option = arguments[i];
            if (option == "--no-ui") no_ui = true;
            else if (option == "--config" || option == "--bag" || option == "--output") {
                if (++i >= arguments.size()) throw std::runtime_error(option + "缺少参数");
                (option == "--config" ? config : option == "--bag" ? bag : output) = arguments[i];
            } else if (option == "--help") {
                std::cout << "run_multi_layer_ndt [--config file.yaml] [--bag bag_directory] [--output map_directory] [--no-ui]\n";
                rclcpp::shutdown(); return 0;
            } else throw std::runtime_error("未知参数: " + option);
        }
        if (config.empty()) config = ament_index_cpp::get_package_share_directory("multi_layer_ndt") + "/config/cx16.yaml";
        auto node = std::make_shared<lightning::MappingNode>(config, output, no_ui);
        if (bag.empty()) node->Subscribe();
        rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
        executor.add_node(node);
        std::thread callbacks([&]() {
            try { executor.spin(); }
            catch (const std::exception& error) {
                RCLCPP_ERROR(node->get_logger(), "ROS执行器失败: %s", error.what()); rclcpp::shutdown();
            }
        });
        int status = 0;
        try {
            if (!bag.empty()) {
                lightning::RosbagIO(bag)
                    .AddPointCloud2Handle(node->LidarTopic(), [&](sensor_msgs::msg::PointCloud2::SharedPtr cloud) {
                        if (!rclcpp::ok() || !node->Alive()) return false;
                        node->ProcessCloud(cloud); return true;
                    }).Go();
                node->Finish();
                if (!node->Save()) status = 1;
            }
            // 离线读完后保留UI供检查；无UI时直接退出。在线始终由订阅回调计算。
            while (status == 0 && rclcpp::ok() && node->Alive() && (bag.empty() || node->HasUI()))
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
        } catch (const std::exception& error) { RCLCPP_ERROR(node->get_logger(), "%s", error.what()); status = 1; }
        rclcpp::shutdown();  // 同时覆盖执行器线程尚未进入spin的退出情况。
        executor.cancel(); callbacks.join();
        if (bag.empty()) {
            node->Finish();
            if (!node->Save()) status = 1;
        }
        executor.remove_node(node); node.reset();
        rclcpp::shutdown();
        return status;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        rclcpp::shutdown(); return 1;
    }
}
