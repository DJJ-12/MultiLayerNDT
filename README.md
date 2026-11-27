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

保存到`--output`指定目录，默认使用`output.save_path`。使用与lightning-lm相同的地图目录结构。

- `global.pcd` 为按图优化后的关键帧位姿叠加并按`output.save_voxel`降采样的全图，默认0.1米，采用压缩二进制PCD。
- `BlockMap/pointcloud_map/seg_X_Y.pcd` 为按XY切分的XYZ点云，`BlockMap/pointcloud_map_metadata.yaml`记录块边长和各块的地图坐标原点。块边长`output.block_map_resolution`默认20米，块内降采样`output.block_map_voxel_size`默认0.1米，设为0时使用全图降采样后的点。负坐标按向下取整确定块原点。
- `pose.txt` 记录图优化后的关键帧位姿，含编号、时间戳、平移和四元数。`trajectory.txt`记录应用图校正后的逐帧位姿，采用TUM格式。

重复保存时更新索引并移除它原先引用但本次已无点的块文件。
保存时只短暂复制关键帧指针/位姿快照，磁盘写入期间在线计算仍可继续。

## 联合多层目标

`ndt.num_layers`人工指定每套体素金字塔的层数，使用`resolutions`和`weights`的前N项；增加层数时需要补齐两份列表。默认N=3，对应2.0、1.0、0.5m，保持原全量模式配置。

全量模式每轮所有有效层在同一个位姿下求值，累加同一套权重的score、gradient和Hessian，再求解一个六维增量。
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

## 可选corner/surface特征链路

原有五份配置默认`features.enabled: false`，使用全量点云链路，只建立一套N层金字塔。设置为`true`后使用从原项目`featureExtraction.cpp`移植的LIO-SAM特征流程，输入必须含PointCloud2的`ring`字段。专用`20261029.yaml`默认开启该模式。

1. 按ring和水平角投影到`N_SCAN × Horizon_SCAN`距离图，同一格保留第一点，按列展开每条ring。
2. 使用前后各5点的深度差平方计算曲率：`c[i] = (sum(r[i-5:i]) + sum(r[i+1:i+6]) - 10*r[i])²`。
3. 沿用0.3m遮挡跳变检查、2%深度差检查，标记不适合选作corner/平坦种子的点。
4. 每条ring分6段，各段按曲率排序，选最多20个corner，选中后抑制前后各5点；抑制遇到大于10列的缺口停止。
5. 按原实现标记低曲率种子，并将全部非corner点收集为surface，再逐ring体素降采样。surface包括未标记点，不是只保留`label=-1`的少量平坦种子。

输入适配保留原始深度用于曲率，输出使用去畸变坐标；surface经过原流程的体素滤波后是各格质心。窗口限制在单条ring中，排序覆盖分区末端，修正跨ring和闭区间边界问题。

```yaml
features:
  enabled: true
  exact_hessian: true
  N_SCAN: 16
  Horizon_SCAN: 2000
  edgeThreshold: 1.0
  surfThreshold: 0.1
  odometrySurfLeafSize: 0.4
  neighbor_radius_factor: 2.0
  corner_distance_scale: 0.3
  surface_distance_scale: 0.2
```

曲率阈值单位为m²，不再读取上一版归一化高/低通的窗口及阈值。`edgeThreshold`降低可增加corner候选，`surfThreshold`控制平坦种子标记；`odometrySurfLeafSize`控制按ring输出的surface密度。`N_SCAN`和`Horizon_SCAN`沿用各原始配置的扫描参数，专用配置按实包填写。

两类点各建立独立的N层体素金字塔：corner只查询corner地图，surface只查询surface地图。特征模式的体素保存占用格内的点云质心，不要求单格至少6个点，也不使用体素高斯协方差。关闭特征开关后，全量NDT仍沿用原来的体素统计、逆协方差、评分与求导。

每个源特征点在各层最多查询5个邻近质心，统一使用`neighbor_radius_factor * 最细启用层的体素边长`作为拟合邻域半径。跨层重合质心合并，至少有5个不同支持才拟合。各层总支持权重沿用`ndt.weights`，在该层找到的质心间均分。corner用加权PCA，最大特征值必须大于第二特征值的3倍。surface以加权质心为局部原点，使用带列主元的QR拟合平面法向，支持点到平面距离须不超过0.2m。拟合不成立只跳过该点。

对应成立后使用几何距离平方的负指数，h由上述两个`distance_scale`参数给出，单位m：

```text
corner:  d² = (T*p-center)' * (I-v*v') * (T*p-center)
surface: d² = (n'*T*p+pd)²
score = exp(-d²/(2*h²))
F = 0.5 * sum(corner_score)/N_corner + 0.5 * sum(surface_score)/N_surface
```

平面偏置pd等于负的法向量与加权质心内积。每个源点的跨层几何分数只计一次，分母使用固定源点数；未匹配点贡献0。单点分数为(0,1]，联合分数为[0,1]，表示几何匹配程度。h越小，大残差衰减越快，但初值偏差较大时梯度也更容易接近0。

几何投影矩阵除以h²后传入原`updateDerivatives`，该路径令`gauss_d1=-1、gauss_d2=1`。梯度与指数评分保持一致。`features.exact_hessian: true` 时使用精确Hessian的三个项求解位姿增量。Newton方向无法上升时，用同一个Hessian的奇异值缩放梯度。设为`false`时固定本轮指数权重，使用Gauss–Newton做对照。两种模式都按指数评分回溯，求解和回溯固定本轮线、面，接受位姿后下一轮重新查询和拟合。
特征路径还把投影二次型的浮点负舍入截到0，防止精确共线时指数略大于1，被原NDT数值检查误丢弃；全量路径不启用此处理。

特征只用于前端NDT匹配。关键帧保存两类局部特征供子地图构建，回环校正时按新位姿重建两类金字塔；后端回环、UI和完整PCD仍使用完整关键帧点云。

特征模式使用过去三个帧间隔的平均运动来估计预测和去畸变所需的速度。历史不足时以最近位姿作为初值，帧内运动速度取零。

没有有效观测时按原运动预测传播，不自动切回全量匹配。日志`源特征(C/S)`是经`ndt.source_voxel`过滤后的点数；`有效线面对应(C/S)`是通过PCA/QR并实际参与本轮评分的源点数。`各层评分`和`有效体素`先列corner的N层，再列surface的N层；特征模式的各层评分是按参与支持权重分摊的几何分数，全部相加再乘0.5等于联合评分，并非每层重复添加一次约束。

### 20260329数据包

`config/20260329.yaml`使用包内的`/cx/cloud`、16个ring和2000列。点内`time`为帧内相对秒。该配置显式启用精确Hessian做实验。将`features.exact_hessian`设为`false`即可对照Gauss–Newton，其余参数保持一致。

### 20261029数据包

已只读抽查该包0、30、300、900、1600秒附近的5帧：topic为`/lidar_points_back`，ring范围0~15，水平角间隔约0.18度，time是0~约0.1秒的帧内相对时间。因此`config/20261029.yaml`使用16行、2000列、`time_scale: 1.0`、`time_absolute: false`；不适用禾赛128线/绝对时间的`my_mapping.yaml`设置。

```bash
ros2 run multi_layer_ndt run_multi_layer_ndt --config "$(ros2 pkg prefix multi_layer_ndt)/share/multi_layer_ndt/config/20261029.yaml" --bag /你的路径/20261029 --output /home/d/maps/20261029_ndt
```

抽帧已运行新提取代码，分别得到723、861、1061、900、1036个corner。该检查与本地回归测试的PCL体素过滤使用替身，不等于完整ROS2/PCL回放；整包地图质量与真实耗时尚未验证。
上述5帧还各抽取约100个corner和100个surface检查跨层几何对应，均得到有效的PCA线与QR面约束。回归覆盖轴对齐及倾斜线/面的解析导数、负指数闭式分数、自由方向、已知位姿恢复、稀疏体素、跨层去重、并行计算及开关关闭后的原NDT路径。

## 位姿图后端与回环

`core/backend/pose_graph.cpp`提取原`addOdomFactor / addLoopFactor / saveKeyFramesAndFactor / correctPoses`中的纯位姿图部分：首帧先验、相邻关键帧BetweenFactor、回环BetweenFactor、iSAM2增量更新，回环后额外更新5次并写回全部关键帧。
原有里程计因子的方差继续使用旋转`1e-6 rad²`、平移`1e-4 m²`；当前纯激光程序把首帧六自由度固定为地图原点，没有引入原LIO-SAM的IMU或GPS因子。

`core/backend/loop.cpp`沿用原来的距离/时间候选、历史邻域叠加、ICP验证与回环相对位姿方向。因为NDT前端没有LOAM角点/面点，回环使用降采样后的完整关键帧。历史目标排除当前帧及时间过近的帧，避免把自身点云算成回环。ICP噪声沿用原实现，以fitness建立对角方差，零误差保留数值下限。

ICP在独立线程运行，只接收不可变关键帧快照并提交回环约束；iSAM2始终在建图调用线程更新。`loopClosureFrequency`按雷达时间限频，离线快速回放也会触发检查；待检查快照保留最新一份。读包结束强制检查末端并等待完成，回环无需等下一个关键帧才能入图。

回环校正不会改写连续odom运动历史。`map<-odom`校正用于下一帧NDT初值和当前输出，防止把图优化跳变当成车辆速度。活动子地图及各层NDT体素立即失效重建；UI重画校正后的轨迹、关键帧和局部地图。逐帧历史轨迹在相邻关键帧之间插值应用SE3校正，关键帧时刻与后端结果一致。

所有配置的`loop`段沿用原参数名。`loopClosureEnableFlag: false`关闭回环检测，保留里程计因子图，方便对照。日志`[后端][回环]`表示ICP通过，`[后端][回环入图]`和`[后端][图优化]`表示约束已用于后端；`[联合NDT]`同时显示累计回环数。

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
| `20261029.yaml` | `/lidar_points_back` | 16线，0~0.1s相对秒 |

禾赛原文件的`time_scale: 1000`不适用于这里直接换算为秒的接收层，已设置`time_scale: 1.0`与`time_absolute: true`。原LOAM特征、IMU、GPS和base外参不参与这个纯激光程序。

## UI与ROS输出

Pangolin中红色为逐帧里程计轨迹，紫色为关键帧轨迹，灰色为当前局部子地图，扫描点按高度着色。
UI只显示活动关键帧以节省渲染内存。导出的`global.pcd`包含全部关键帧的地图点云。

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
特征测试覆盖16线平面/折角、ring/列投影、曲率阈值、遮挡、六分区每区20角点上限、邻域抑制、同格第一回波、ring边界、surface收集和体素滤波，以及独立金字塔、联合解析导数、显式层数与两条建图链路。
Windows开发环境没有ROS2/PCL，完整Ubuntu构建与真实bag回放需要在ROS2机器验证。
本地已用真实Eigen/Sophus和复制的NDT导数代码编译、运行新建图核心测试；PCL体素、邻域和PCD读写在该隔离测试中使用替身，因此不替代完整ROS2/PCL测试。
后端已额外编译并链接真实GTSAM进行验证：人工正方形末端漂移由0.8m降至约0.00017m，原连续里程计保持不变；异步回环链路测试中的PCL/ICP仍使用替身，真实数据效果与耗时尚未验证。
