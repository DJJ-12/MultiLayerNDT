#include "wrapper/mapping_node.h"

#include <yaml-cpp/yaml.h>
#include <pcl_conversions/pcl_conversions.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace lightning {
namespace {
template<class T> double Read(const std::uint8_t* source, bool reverse) {
    std::uint8_t bytes[sizeof(T)];
    std::memcpy(bytes, source, sizeof(T));
    if (reverse) std::reverse(bytes, bytes + sizeof(T));
    T value; std::memcpy(&value, bytes, sizeof(T));
    return static_cast<double>(value);
}
double ReadField(const std::uint8_t* data, const sensor_msgs::msg::PointField& field, bool reverse) {
    using F = sensor_msgs::msg::PointField;
    const auto* source = data + field.offset;
    switch (field.datatype) {
        case F::INT8: return Read<std::int8_t>(source, reverse);
        case F::UINT8: return Read<std::uint8_t>(source, reverse);
        case F::INT16: return Read<std::int16_t>(source, reverse);
        case F::UINT16: return Read<std::uint16_t>(source, reverse);
        case F::INT32: return Read<std::int32_t>(source, reverse);
        case F::UINT32: return Read<std::uint32_t>(source, reverse);
        case F::FLOAT32: return Read<float>(source, reverse);
        case F::FLOAT64: return Read<double>(source, reverse);
        default: throw std::runtime_error("不支持的PointCloud2字段类型");
    }
}
geometry_msgs::msg::Pose ToMessage(const SE3& pose) {
    geometry_msgs::msg::Pose message;
    const auto& t = pose.translation(); const auto q = pose.unit_quaternion();
    message.position.x = t.x(); message.position.y = t.y(); message.position.z = t.z();
    message.orientation.x = q.x(); message.orientation.y = q.y();
    message.orientation.z = q.z(); message.orientation.w = q.w();
    return message;
}
}  // namespace

MappingNode::MappingNode(const std::string& yaml_path, const std::string& save_path, bool no_ui)
    : rclcpp::Node("multi_layer_ndt") {
    const auto yaml = YAML::LoadFile(yaml_path);
    const auto sensor = yaml["sensor"], ndt = yaml["ndt"], map = yaml["mapping"], output = yaml["output"];
    lidar_topic_ = sensor["lidar_topic"].as<std::string>();
    time_field_ = sensor["time_field"].as<std::string>();
    time_scale_ = sensor["time_scale"].as<double>();
    time_absolute_ = sensor["time_absolute"].as<bool>(false);
    deskew_ = sensor["deskew"].as<bool>();
    min_range_ = sensor["min_range"].as<double>(); max_range_ = sensor["max_range"].as<double>();
    MappingOptions options;
    options.ndt.resolutions = ndt["resolutions"].as<std::vector<double>>();
    options.ndt.weights = ndt["weights"].as<std::vector<double>>();
    options.ndt.num_layers = ndt["num_layers"].as<int>(static_cast<int>(options.ndt.resolutions.size()));
    options.ndt.source_voxel = ndt["source_voxel"].as<double>();
    options.ndt.max_iterations = ndt["max_iterations"].as<int>();
    options.ndt.num_threads = ndt["num_threads"].as<int>();
    options.ndt.epsilon = ndt["epsilon"].as<double>(); options.ndt.max_step = ndt["max_step"].as<double>();
    options.keyframe_distance = map["keyframe_distance"].as<double>();
    options.keyframe_angle = map["keyframe_angle"].as<double>();
    options.keyframe_time = map["keyframe_time"].as<double>();
    options.keyframe_voxel = map["keyframe_voxel"].as<double>();
    options.submap_radius = map["submap_radius"].as<double>();
    options.submap_keyframes = map["submap_keyframes"].as<int>();
    if (const auto features = yaml["features"]) {
        options.features.enabled = features["enabled"].as<bool>(false);
        options.features.exact_hessian = features["exact_hessian"].as<bool>(options.features.exact_hessian);
        options.features.use_jacobian_term = features["use_jacobian_term"].as<bool>(options.features.use_jacobian_term);
        options.features.hessian_constraint_weight = features["hessian_constraint_weight"].as<double>(0.0);
        options.features.N_SCAN = features["N_SCAN"].as<int>(16);
        options.features.Horizon_SCAN = features["Horizon_SCAN"].as<int>(2000);
        options.features.edgeThreshold = features["edgeThreshold"].as<float>(1.0f);
        options.features.surfThreshold = features["surfThreshold"].as<float>(0.1f);
        options.features.odometrySurfLeafSize = features["odometrySurfLeafSize"].as<float>(0.4f);
        options.features.neighbor_radius_factor = features["neighbor_radius_factor"].as<double>(2.0);
        options.features.corner_distance_scale = features["corner_distance_scale"].as<double>(0.3);
        options.features.surface_distance_scale = features["surface_distance_scale"].as<double>(0.2);
    }
    use_features_ = options.features.enabled;
    if (const auto loop = yaml["loop"]) {
        options.loop.loopClosureEnableFlag = loop["loopClosureEnableFlag"].as<bool>(true);
        options.loop.loopClosureFrequency = loop["loopClosureFrequency"].as<double>(1.0);
        options.loop.historyKeyframeSearchRadius = loop["historyKeyframeSearchRadius"].as<double>(15.0);
        options.loop.historyKeyframeSearchTimeDiff = loop["historyKeyframeSearchTimeDiff"].as<double>(30.0);
        options.loop.historyKeyframeSearchNum = loop["historyKeyframeSearchNum"].as<int>(25);
        options.loop.historyKeyframeFitnessScore = loop["historyKeyframeFitnessScore"].as<double>(0.3);
        options.loop.mappingSurfLeafSize = loop["mappingSurfLeafSize"].as<double>(0.4);
    }
    mapping_ = std::make_unique<Mapping>(options);
    map_frame_ = output["map_frame"].as<std::string>(); lidar_frame_ = output["lidar_frame"].as<std::string>();
    save_path_ = save_path.empty() ? output["save_path"].as<std::string>() : save_path;
    save_voxel_ = output["save_voxel"].as<double>(); pub_tf_ = output["pub_tf"].as<bool>();
    block_map_resolution_ = output["block_map_resolution"].as<int>(block_map_resolution_);
    block_map_voxel_size_ = output["block_map_voxel_size"].as<double>(block_map_voxel_size_);
    const auto durable = rclcpp::QoS(1).reliable().transient_local();
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/multi_layer_ndt/pose", 10);
    path_pub_ = create_publisher<nav_msgs::msg::Path>("/multi_layer_ndt/path", durable);
    map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/multi_layer_ndt/map", durable);
    scan_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/multi_layer_ndt/scan", rclcpp::SensorDataQoS().keep_last(1));
    tf_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);
    service_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    save_service_ = create_service<std_srvs::srv::Trigger>("/multi_layer_ndt/save_map",
        [this](const std_srvs::srv::Trigger::Request::SharedPtr, std_srvs::srv::Trigger::Response::SharedPtr response) {
            response->success = Save();
            response->message = response->success ? save_path_ : "地图为空或保存失败";
        }, rmw_qos_profile_services_default, service_group_);
#ifdef MULTI_LAYER_NDT_WITH_UI
    if (!no_ui && output["with_ui"].as<bool>()) {
        ui_ = std::make_shared<ui::PangolinWindow>();
        if (!ui_->Init()) ui_.reset();
        if (ui_) ui_->SetCurrentScanSize(3);
    }
#else
    (void)no_ui;
#endif
    const char* mode = "全量点云";
    if (use_features_) mode = options.features.exact_hessian ? "特征/精确Hessian" : "特征/Gauss-Newton";
    RCLCPP_INFO(get_logger(), "点云入口=%s，输出目录=%s，每套金字塔层数=%d，模式=%s，线面线性项=%s", lidar_topic_.c_str(),
        save_path_.c_str(), options.ndt.num_layers, mode, options.features.UseJointSolve() ? "开启" : "关闭");
}
MappingNode::~MappingNode() {
#ifdef MULTI_LAYER_NDT_WITH_UI
    if (ui_) ui_->Quit();
#endif
}

void MappingNode::Subscribe() {
    subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(lidar_topic_,
        rclcpp::SensorDataQoS().keep_last(1), [this](sensor_msgs::msg::PointCloud2::SharedPtr cloud) { ProcessCloud(cloud); });
}

Scan MappingNode::Convert(const sensor_msgs::msg::PointCloud2& message) const {
    const auto field = [&](const std::string& name) -> const sensor_msgs::msg::PointField* {
        for (const auto& item : message.fields) if (item.name == name) return &item;
        return nullptr;
    };
    const auto* x = field("x"); const auto* y = field("y"); const auto* z = field("z");
    const auto* intensity = field("intensity"); const auto* time = deskew_ ? field(time_field_) : nullptr;
    const auto* ring = use_features_ ? field("ring") : nullptr;
    if (!x || !y || !z) throw std::runtime_error("点云缺少x/y/z字段");
    if (use_features_ && !ring) throw std::runtime_error("features.enabled=true时点云必须包含ring字段");
    // 同时支持有row padding的有序点云及大小端数据，不假定time一定是float。
    const std::uint16_t probe = 1;
    const bool host_big = *reinterpret_cast<const std::uint8_t*>(&probe) == 0;
    const bool reverse = host_big != message.is_bigendian;
    Scan scan;
    scan.stamp = rclcpp::Time(message.header.stamp).seconds();
    scan.cloud.reset(new PointCloudType);
    scan.cloud->reserve(static_cast<std::size_t>(message.width) * message.height);
    if (time) scan.relative_times.reserve(scan.cloud->points.capacity());
    if (ring) scan.rings.reserve(scan.cloud->points.capacity());
    for (std::uint32_t row = 0; row < message.height; ++row) for (std::uint32_t col = 0; col < message.width; ++col) {
        const auto offset = static_cast<std::size_t>(row) * message.row_step + static_cast<std::size_t>(col) * message.point_step;
        if (offset + message.point_step > message.data.size()) throw std::runtime_error("点云data长度与width/row_step不一致");
        const auto* data = message.data.data() + offset;
        PointType point;
        point.x = ReadField(data, *x, reverse); point.y = ReadField(data, *y, reverse); point.z = ReadField(data, *z, reverse);
        const double distance = point.getVector3fMap().norm();
        if (!std::isfinite(distance) || distance < min_range_ || distance > max_range_) continue;
        point.intensity = intensity ? ReadField(data, *intensity, reverse) : 0.0;
        if (!std::isfinite(point.intensity)) point.intensity = 0.0;
        scan.cloud->push_back(point);
        if (ring) scan.rings.push_back(static_cast<std::uint16_t>(ReadField(data, *ring, reverse)));
        if (time) {
            const double relative = time_scale_ * ReadField(data, *time, reverse) - (time_absolute_ ? scan.stamp : 0.0);
            scan.relative_times.push_back(std::isfinite(relative) ? relative : 0.0);
        }
    }
    return scan;
}

void MappingNode::ProcessCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud) {
    try {
        const auto begin = std::chrono::steady_clock::now();
        const Scan scan = Convert(*cloud);
        if (scan.cloud->empty()) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "点云经距离/非有限点过滤后为空");
            return;
        }
        std::lock_guard<std::mutex> lock(processing_mutex_);
        const auto result = mapping_->Process(scan);
        if (!result.valid) return;
        last_header_ = cloud->header;
        if (frames_ == 0) {
            RCLCPP_INFO(get_logger(), "首帧有效点=%zu，点内时间=%s，第一帧建立map原点",
                scan.cloud->size(), scan.relative_times.empty() ? "无/未启用" : "已转换为相对秒");
            if (!scan.rings.empty()) {
                const auto range = std::minmax_element(scan.rings.begin(), scan.rings.end());
                RCLCPP_INFO(get_logger(), "[LIO-SAM特征] 收到ring范围=%u..%u", unsigned(*range.first), unsigned(*range.second));
            }
        }
        Publish(result, cloud->header);
        if (++frames_ % 20 == 1) {
            std::ostringstream scores; for (double score : result.layer_scores) scores << score << ' ';
            std::ostringstream voxels; for (auto count : result.layer_voxels) voxels << count << ' ';
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            RCLCPP_INFO(get_logger(), "[联合NDT] 模式=%s 帧=%zu 关键帧=%zu 回环=%zu 迭代=%d 收敛=%d 源特征(C/S)=%zu/%zu 有效线面对应(C/S)=%zu/%zu 各层评分=[%s] 有效体素=[%s] 总耗时=%.2fms",
                use_features_ ? "PCA/QR几何指数" : "full", frames_, result.keyframe_count, result.loop_count, result.iterations,
                int(result.converged), result.corner_count, result.surface_count, result.corner_matches, result.surface_matches,
                scores.str().c_str(), voxels.str().c_str(), ms);
        }
    } catch (const std::exception& error) { RCLCPP_ERROR(get_logger(), "点云处理失败: %s", error.what()); }
}

void MappingNode::Publish(const MappingResult& result, const std_msgs::msg::Header& header) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header = header; pose.header.frame_id = map_frame_; pose.pose = ToMessage(result.state.pose);
    pose_pub_->publish(pose);
    if (pub_tf_) {
        geometry_msgs::msg::TransformStamped transform;
        transform.header = pose.header; transform.child_frame_id = lidar_frame_;
        transform.transform.translation.x = pose.pose.position.x;
        transform.transform.translation.y = pose.pose.position.y;
        transform.transform.translation.z = pose.pose.position.z;
        transform.transform.rotation = pose.pose.orientation;
        tf_->sendTransform(transform);
    }
    if (result.graph_updated) {
        path_.poses.clear();
        for (const auto& frame : mapping_->Data().keyframes) {
            auto key_pose = pose;
            key_pose.header.stamp = rclcpp::Time(static_cast<std::int64_t>(std::llround(frame.stamp * 1e9)));
            key_pose.pose = ToMessage(frame.pose); path_.poses.push_back(key_pose);
        }
    }
    else if (result.keyframe_added) path_.poses.push_back(pose);
    if (result.keyframe_added || result.graph_updated) { path_.header = pose.header; path_pub_->publish(path_); }
    if (result.scan && scan_pub_->get_subscription_count()) {
        sensor_msgs::msg::PointCloud2 message;
        pcl::toROSMsg(*result.scan, message); message.header = header; message.header.frame_id = lidar_frame_;
        scan_pub_->publish(message);
    }
    if (result.map_updated) {
        PointCloudType map;
        for (const auto& item : result.local_map) map += *item.second;
        sensor_msgs::msg::PointCloud2 message;
        pcl::toROSMsg(map, message); message.header = pose.header; map_pub_->publish(message);
    }
#ifdef MULTI_LAYER_NDT_WITH_UI
    if (ui_) {
        ui_->UpdateNavState(result.state);
        if (result.graph_updated) ui_->UpdateTrajectory(Mapping::CorrectedTrajectory(mapping_->Data()));
        if (result.scan) ui_->UpdateScan(result.scan, result.state.pose);
        if (result.map_updated) ui_->UpdatePointCloudGlobal(result.local_map, result.graph_updated);
        if (result.keyframe_added || result.graph_updated) {
            std::vector<NavState> poses;
            for (const auto& frame : mapping_->Data().keyframes) { NavState state; state.pose = frame.pose; poses.push_back(state); }
            ui_->UpdateKeyframePoses(poses);
        }
    }
#endif
}

void MappingNode::Finish() {
    std::lock_guard<std::mutex> processing(processing_mutex_);
    const auto result = mapping_->Finish();
    if (result.valid && rclcpp::ok()) Publish(result, last_header_);
}

bool MappingNode::Save() {
    std::lock_guard<std::mutex> saving(saving_mutex_);
    try {
        MapData snapshot;
        { std::lock_guard<std::mutex> processing(processing_mutex_); snapshot = mapping_->Data(); }
        const bool saved = Mapping::SaveMap(snapshot, save_path_, save_voxel_, block_map_resolution_, block_map_voxel_size_);
        if (saved) RCLCPP_INFO(get_logger(), "已保存地图与轨迹: %s", save_path_.c_str());
        return saved;
    } catch (const std::exception& error) { RCLCPP_ERROR(get_logger(), "地图保存失败: %s", error.what()); return false; }
}
bool MappingNode::HasUI() const {
#ifdef MULTI_LAYER_NDT_WITH_UI
    return static_cast<bool>(ui_);
#else
    return false;
#endif
}
bool MappingNode::Alive() {
#ifdef MULTI_LAYER_NDT_WITH_UI
    return !ui_ || !ui_->ShouldQuit();
#else
    return true;
#endif
}
}  // namespace lightning
