# MultiLayerNDT 轻量建图

这是独立的ROS2包`multi_layer_ndt`，与旁边的`lightning`包互不替换。复用复制的NDT体素统计、邻域查询、解析梯度/Hessian和Pangolin显示。
程序无需预建地图：第一帧为map原点，后续与附近历史关键帧子地图匹配。CPU运行，后端和回环从原LIO-SAM移植，使用GTSAM/iSAM2；无GPS、EKF或预建地图加载依赖。

## 构建

在ROS2工作空间根目录运行（依赖Eigen、PCL、GTSAM、OpenMP、yaml-cpp、glog，以及UI使用的Pangolin/OpenGL；GTSAM沿用原lightning安装）：

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
读完会等待最后一次回环检查、应用图优化，再自动保存；有UI时窗口保留供检查，关闭窗口退出。`--no-ui`用于不打开窗口并在离线结束后直接退出。
在线按Ctrl+C或关闭UI后保存。建图过程中也可以主动保存：

```bash
ros2 service call /multi_layer_ndt/save_map std_srvs/srv/Trigger "{}"
```

保存到`--output`指定目录；未给时使用`output.save_path`。输出`map.pcd`（按图优化后的关键帧位姿叠加并降采样）和`trajectory.txt`（应用关键帧图校正后的全部处理帧，TUM格式：time x y z qx qy qz qw）。再次保存会覆盖这两个输出文件，不清空目录。
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
每层体素只在附近关键帧集合变化或回环校正时重建。历史关键帧只保存自身雷达系的一份点云；活动子地图保留有限的map系变换缓存，UI共享该缓存。
有效但未收敛的结果仍用于位姿传播，暂不加入关键帧；无有效对应时沿运动预测传播。
`converged`是数值收敛，不能保证匹配一定正确，连续退化或错误关键帧仍可能导致漂移。

## LIO-SAM后端与回环

`core/lio_sam/map_optimization.cpp`提取原`addOdomFactor / addLoopFactor / saveKeyFramesAndFactor / correctPoses`中的纯位姿图部分：首帧先验、相邻关键帧BetweenFactor、回环BetweenFactor、iSAM2增量更新，回环后额外更新5次并写回全部关键帧。
原有里程计因子的方差继续使用旋转`1e-6 rad²`、平移`1e-4 m²`；当前纯激光程序把首帧六自由度固定为地图原点，没有引入原LIO-SAM的IMU或GPS因子。

`core/lio_sam/loop.cpp`沿用原来的距离/时间候选、历史邻域叠加、ICP验证与回环相对位姿方向。因为NDT前端没有LOAM角点/面点，回环使用降采样后的完整关键帧。历史目标排除当前帧及时间过近的帧，避免把自身点云算成回环。ICP噪声沿用原实现，以fitness建立对角方差，零误差保留数值下限。

ICP在独立线程运行，只接收不可变关键帧快照并提交回环约束；iSAM2始终在建图调用线程更新。`loopClosureFrequency`按雷达时间限频，离线快速回放也会触发检查；待检查快照保留最新一份。读包结束强制检查末端并等待完成，回环无需等下一个关键帧才能入图。

回环校正不会改写连续odom运动历史。`map<-odom`校正用于下一帧NDT初值和当前输出，防止把图优化跳变当成车辆速度。活动子地图及各层NDT体素立即失效重建；UI重画校正后的轨迹、关键帧和局部地图。逐帧历史轨迹在相邻关键帧之间插值应用SE3校正，关键帧时刻与后端结果一致。

所有配置的`lio_sam`段沿用原参数名。`loopClosureEnableFlag: false`关闭回环检测，保留里程计因子图，方便对照。日志`[LIO-SAM][回环]`表示ICP通过，`[LIO-SAM][回环入图]`和`[LIO-SAM][图优化]`表示约束已用于后端；`[联合NDT]`同时显示累计回环数。

## 点云与时间

配置文件的`sensor.lidar_topic`必须与bag/传感器一致；Campus默认`/Bob/velodyne_points`，CX16默认`/cx/cloud`。
支持x/y/z及可选intensity字段。`time_field`指定点内相对时间，`time_scale`把该字段转换为秒，例如`time`浮点秒用1，`t`整数纳秒用1e-9。
点内时间默认相对header.stamp；绝对秒字段设置`time_absolute: true`，接收层先减去header.stamp。默认使用前两帧的恒速模型去畸变到header.stamp，不需要IMU。缺少时间字段时直接使用点云。
这不是IMU辅助去畸变，高动态/强加速度下需单独评估。所有位姿均表示map<-lidar，第一帧为单位位姿。

补充的配置对应原项目同名文件，保留其topic、量程与回环参数，前端仍使用本程序的联合NDT：

| 配置 | 点云topic | 点内time |
|---|---|---|
| `chaosuan.yaml` | `/cx/cloud` | 相对秒 |
| `geli.yaml` | `/lidar_points_back` | 相对秒 |
| `my_mapping.yaml` | `/lidar_points_back` | 禾赛绝对秒，接收层转换 |

禾赛原文件的`time_scale: 1000`不适用于这里直接换算为秒的接收层，已设置`time_scale: 1.0`与`time_absolute: true`。原LOAM特征、IMU、GPS和base外参不参与这个纯激光程序。

## UI与ROS输出

Pangolin中红色为逐帧里程计轨迹，紫色为关键帧轨迹，灰色为当前局部子地图，扫描点按高度着色。
UI只显示活动关键帧，避免完整历史点云一直占用渲染内存；保存的map.pcd仍包含全部关键帧。

| Topic | 消息 | 内容 |
|---|---|---|
| `/multi_layer_ndt/pose` | geometry_msgs/PoseStamped | 每个处理帧的map系位姿 |
| `/multi_layer_ndt/path` | nav_msgs/Path | 优化后关键帧轨迹，回环时替换全部历史位姿 |
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
后端测试增加人工漂移正方形的回环收敛、首帧锚定、前端运动历史隔离、轨迹校正插值、异步ICP回环、重复回环及时间邻近排除。
Windows开发环境没有ROS2/PCL，完整Ubuntu构建与真实bag回放需要在ROS2机器验证。
本地已用真实Eigen/Sophus和复制的NDT导数代码编译、运行新建图核心测试；PCL体素、邻域和PCD读写在该隔离测试中使用替身，因此不替代完整ROS2/PCL测试。
后端已额外编译并链接真实GTSAM进行验证：人工正方形末端漂移由0.8m降至约0.00017m，原连续里程计保持不变；异步回环链路测试中的PCL/ICP仍使用替身，真实数据效果与耗时尚未验证。
