# MultiLayerNDT 轻量建图

这是独立的ROS2包`multi_layer_ndt`，与旁边的`lightning`包互不替换。复用复制的NDT体素统计、邻域查询、解析梯度/Hessian和Pangolin显示。
程序无需预建地图：第一帧为map原点，后续与附近历史关键帧子地图匹配。CPU运行，无GPS、EKF、GTSAM、回环或预建地图加载依赖。

## 构建

在ROS2工作空间根目录运行（依赖Eigen、PCL、OpenMP、yaml-cpp、glog，以及UI使用的Pangolin/OpenGL）：

```bash
colcon build --packages-select multi_layer_ndt --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

Pangolin可使用原lightning项目已安装的版本。无图形环境可加`-DWITH_UI=OFF`；这时不需要Pangolin和OpenGL。
已有目录里的MapLoader、旧Localization、TPE和协方差代码保留作参考，不参加新程序构建。

## 在线订阅

```bash
ros2 run multi_layer_ndt run_multi_layer_ndt --config /你的路径/MultiLayerNDT/config/cx16.yaml --output /home/d/maps/ndt_test
```

仅订阅`sensor_msgs/msg/PointCloud2`。在线QoS为SensorData、KeepLast(1)，回调内同步计算，不另建生产者/消费者队列；处理中旧的待处理帧会被覆盖。
算法初值按实际时间间隔缩放上一段相对运动，不限制绝对速度。

## 离线直接读bag

```bash
ros2 run multi_layer_ndt run_multi_layer_ndt --config /你的路径/MultiLayerNDT/config/cx16.yaml --bag /home/d/dataset/20260921_01 --output /home/d/maps/ndt_test
ros2 run multi_layer_ndt run_multi_layer_ndt --config /你的路径/MultiLayerNDT/config/s3e_campus_road_1_bob.yaml --bag /home/d/dataset/S3E_Campus_Road_1 --output /home/d/maps/campus_ndt
```

直接读SQLite3 bag，不通过DDS、不跳帧，也不建立在线订阅。只读取配置中的点云topic。
读完自动保存；有UI时窗口保留供检查，关闭窗口退出。`--no-ui`用于不打开窗口并在离线结束后直接退出。
在线按Ctrl+C或关闭UI后保存。建图过程中也可以主动保存：

```bash
ros2 service call /multi_layer_ndt/save_map std_srvs/srv/Trigger "{}"
```

保存到`--output`指定目录；未给时使用`output.save_path`。输出`map.pcd`（完整关键帧地图降采样）和`trajectory.txt`（全部处理帧，TUM格式：time x y z qx qy qz qw）。再次保存会覆盖这两个输出文件，不清空目录。
保存时只短暂复制关键帧指针/位姿快照，磁盘写入期间在线计算仍可继续。

## 联合多层目标

每轮所有有效层在同一个位姿下求值，累加同一套权重的score、gradient和Hessian，再求解一个六维增量。
没有“粗层align完成后再跑细层”，没有把多个独立位姿作平均。每层复用`MultiGridNormalDistributionsTransform::computeDerivatives`。

各层score先除以固定源点数N及本层高斯幅值`A=-gauss_d1`，权重归一化到和为1：

```text
F = sum_l w_l * S_l / (N*A_l)
g = sum_l w_l * g_l / (N*A_l)
H = sum_l w_l * H_l / (N*A_l)
H * delta = -g
```

试探步长始终评价同一个联合目标；非上升Newton方向退回上升方向，有限次回溯选择增益步长。
`ndt.max_step`限制每次优化迭代的增量范数，不限制车辆速度；`epsilon`用于判断迭代增量是否收敛。
每层体素只在附近关键帧集合变化时重建。关键帧map系点云只保存一份，子地图、保存与UI共享它。
有效但未收敛的结果仍用于位姿传播，暂不加入关键帧；无有效对应时沿运动预测传播。
`converged`是数值收敛，不能保证匹配一定正确，连续退化或错误关键帧仍可能导致漂移。

## 点云与时间

配置文件的`sensor.lidar_topic`必须与bag/传感器一致；Campus默认`/Bob/velodyne_points`，CX16默认`/cx/cloud`。
支持x/y/z及可选intensity字段。`time_field`指定点内相对时间，`time_scale`把该字段转换为秒，例如`time`浮点秒用1，`t`整数纳秒用1e-9。
点内时间须相对header.stamp；默认使用前两帧的恒速模型去畸变到header.stamp，不需要IMU。缺少时间字段时直接使用点云。
这不是IMU辅助去畸变，高动态/强加速度下需单独评估。所有位姿均表示map<-lidar，第一帧为单位位姿。

## UI与ROS输出

Pangolin中红色为逐帧里程计轨迹，紫色为关键帧轨迹，灰色为当前局部子地图，扫描点按高度着色。
UI只显示活动关键帧，避免完整历史点云一直占用渲染内存；保存的map.pcd仍包含全部关键帧。

| Topic | 消息 | 内容 |
|---|---|---|
| `/multi_layer_ndt/pose` | geometry_msgs/PoseStamped | 每个处理帧的map系位姿 |
| `/multi_layer_ndt/path` | nav_msgs/Path | 关键帧轨迹 |
| `/multi_layer_ndt/scan` | sensor_msgs/PointCloud2 | 去畸变后的当前帧，lidar系 |
| `/multi_layer_ndt/map` | sensor_msgs/PointCloud2 | 活动关键帧子地图，map系 |

按`output.pub_tf`发布map->lidar。没有给未校准的Hessian逆矩阵贴上“真实位姿协方差”标签。
日志`[联合NDT]`显示各层归一化评分、迭代数、关键帧数和含子地图更新的处理耗时。

## 回归测试

```bash
colcon build --packages-select multi_layer_ndt --cmake-args -DBUILD_TESTING=ON -DWITH_UI=OFF
colcon test --packages-select multi_layer_ndt --event-handlers console_direct+
colcon test-result --verbose
```

测试覆盖各层共同贡献、联合梯度/Hessian与分层加权的一致性、解析导数数值差分、已知位姿配准、第一帧原点、重复时间戳及地图/轨迹保存。
Windows开发环境没有ROS2/PCL，完整Ubuntu构建与真实bag回放需要在ROS2机器验证。
本地已用真实Eigen/Sophus和复制的NDT导数代码编译、运行新建图核心测试；PCL体素、邻域和PCD读写在该隔离测试中使用替身，因此不替代完整ROS2/PCL测试。
