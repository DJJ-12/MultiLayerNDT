add_library(multi_layer_ndt_core STATIC
  core/mapping/multi_layer_ndt.cc
  core/mapping/mapping.cc
  core/mapping/save_map.cc
  core/mapping/feature_extraction.cc
  core/backend/pose_graph.cpp
  core/backend/loop.cpp
  core/localization/GlobalLocalizer/ndt_omp/multigrid_ndt_omp.cpp
  core/localization/GlobalLocalizer/ndt_omp/multi_voxel_grid_covariance_omp.cpp)
target_include_directories(multi_layer_ndt_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/core/localization
  ${PROJECT_SOURCE_DIR}/thirdparty ${PCL_INCLUDE_DIRS} ${GTSAM_INCLUDE_DIR})
target_compile_definitions(multi_layer_ndt_core PUBLIC PCL_NO_PRECOMPILE SOPHUS_USE_BASIC_LOGGING)
target_link_libraries(multi_layer_ndt_core PUBLIC Eigen3::Eigen ${PCL_LIBRARIES} OpenMP::OpenMP_CXX
  gtsam glog::glog Threads::Threads)
add_executable(run_multi_layer_ndt app/main.cc wrapper/mapping_node.cc wrapper/bag_io.cc)
ament_target_dependencies(run_multi_layer_ndt rclcpp ament_index_cpp sensor_msgs std_msgs
  geometry_msgs nav_msgs std_srvs tf2_ros pcl_conversions rosbag2_cpp rosbag2_storage)
target_link_libraries(run_multi_layer_ndt multi_layer_ndt_core yaml-cpp glog::glog Threads::Threads)
if(WITH_UI)
  target_sources(run_multi_layer_ndt PRIVATE ui/pangolin_window.cc ui/pangolin_window_impl.cc
    ui/ui_cloud.cc ui/ui_car.cc ui/ui_trajectory.cc)
  target_include_directories(run_multi_layer_ndt PRIVATE ${Pangolin_INCLUDE_DIRS})
  target_compile_definitions(run_multi_layer_ndt PRIVATE MULTI_LAYER_NDT_WITH_UI)
  target_link_libraries(run_multi_layer_ndt ${Pangolin_LIBRARIES} OpenGL::GL)
endif()
install(TARGETS run_multi_layer_ndt DESTINATION lib/${PROJECT_NAME})
if(BUILD_TESTING)
  add_executable(multi_layer_ndt_test ../tests/multi_layer_ndt_test.cc)
  target_link_libraries(multi_layer_ndt_test multi_layer_ndt_core)
  add_test(NAME multi_layer_ndt_core COMMAND multi_layer_ndt_test)
  add_executable(multi_layer_ndt_backend_test ../tests/backend_test.cc)
  target_link_libraries(multi_layer_ndt_backend_test multi_layer_ndt_core)
  add_test(NAME multi_layer_ndt_backend COMMAND multi_layer_ndt_backend_test)
  add_executable(multi_layer_ndt_feature_test ../tests/feature_test.cc)
  target_link_libraries(multi_layer_ndt_feature_test multi_layer_ndt_core)
  add_test(NAME multi_layer_ndt_features COMMAND multi_layer_ndt_feature_test)
endif()
