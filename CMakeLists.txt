cmake_minimum_required(VERSION 2.8.3)
project(lidar_track)

##Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED
pcl_ros
roscpp
rviz_visual_tools
sensor_msgs
std_msgs
pcl_conversions
rospy
tf
cv_bridge
image_transport
compressed_image_transport
jsk_recognition_msgs
jsk_rviz_plugins
tf2
tf2_ros
tf2_msgs
argo_track_msgs
track_array_msgs
argo_detection

)



catkin_package(
   CATKIN_DEPENDS 
	   pcl_ros
	   roscpp
	   sensor_msgs
	   pcl_conversions
   INCLUDE_DIRS include
#  LIBRARIES track
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)



## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
 include
 src
 ${catkin_INCLUDE_DIRS}
 ${PCL_INCLUDE_DIRS}
)


## Declare a C++ library
add_executable(
   kal
   src/modified_kf_track.cpp src/Hungarian.cpp
 )

add_executable(
   test_a
   src/see.cpp)



add_executable(
   test_hun
   src/hungarian_test.cpp src/Hungarian.cpp)


add_executable(
   argo_test
   src/argo_tracking_test_viz.cpp)

add_executable(
   kal_gt
   src/modified_kf_track_gt.cpp src/Hungarian.cpp
 )

add_executable(
   test_csv
   src/write_csv_test.cpp
 )

add_executable(
   gt_output
   src/gt_output.cpp
)

add_executable(
   region_seg
   src/region_growing_seg.cpp
)

add_executable(
   viz_detection
   src/test_detection.cpp
)

add_executable(
   kal_update
   src/kf_track_update.cpp src/Hungarian.cpp
 )

add_executable(
   image_read
   src/image_read.cpp
 )


add_executable(
   subscribe_compressed_img
   src/subscribe_compressed_img.cpp
 )

add_executable(
   img_diff
   src/img_diff.cpp
 )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/track_node.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
 target_link_libraries(
   kal
   ${catkin_LIBRARIES}
 )
 target_link_libraries(
   test_a
   ${catkin_LIBRARIES}
 )
target_link_libraries(
   test_hun
   ${catkin_LIBRARIES}
 )

target_link_libraries(
   argo_test
   ${catkin_LIBRARIES}
 )

target_link_libraries(
   kal_gt
   ${catkin_LIBRARIES}
 )

target_link_libraries(
   test_csv
   ${catkin_LIBRARIES}
 )

target_link_libraries(
   gt_output
   ${catkin_LIBRARIES}
 )

target_link_libraries(
   region_seg
   ${catkin_LIBRARIES}
   ${PCL_LIBRARIES}
 )

target_link_libraries(
   viz_detection
   ${catkin_LIBRARIES}
   ${PCL_LIBRARIES}
 )

target_link_libraries(
   kal_update
   ${catkin_LIBRARIES}
 )

target_link_libraries(
   image_read
   ${catkin_LIBRARIES}
 )

target_link_libraries(
   subscribe_compressed_img
   ${catkin_LIBRARIES}
   ${PCL_LIBRARIES}
   ${OpenCV_LIBRARIES}
 )

target_link_libraries(
   img_diff
   ${catkin_LIBRARIES}
   ${PCL_LIBRARIES}
   ${OpenCV_LIBRARIES}
 )




add_dependencies(gt_output argo_track_msgs_gencpp track_array_msgs_gencpp ${catkin_EXPORTED_TARGETS})


