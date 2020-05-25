#include <iostream>
#include <stdlib.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <pcl/conversions.h>
#include <pcl/filters/passthrough.h>
#include <tf/transform_broadcaster.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include "pcl_ros/point_cloud.h"
#include <pcl/impl/point_types.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/icp.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <pcl/filters/voxel_grid.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Point.h>
#include <iostream>
#include <tf/LinearMath/Matrix3x3.h>
#include <tf/LinearMath/Vector3.h>
//#include <localization/gps_transformer.h>	
#include <string> 
#include <fstream>
#include <sstream>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/console/time.h>


#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>


#include <algorithm>
#include <iterator>
#include <math.h>
#include "kf_tracker/featureDetection.h"
#include "kf_tracker/CKalmanFilter.h"
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Int32MultiArray.h>


#include "Hungarian/Hungarian.h"
// #include "Hungarian/Hungarian.cpp"


#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/geometry.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h>
 
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <limits>
#include <utility>

#include <tf/LinearMath/Matrix3x3.h>
#include <tf/LinearMath/Vector3.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

#include "Eigen/Eigen"
#include <Eigen/Geometry>

//get tf to tranform to global
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2/convert.h>
#include <tf2_ros/buffer.h>
#include <tf2/transform_datatypes.h>
// #include <tf2_sensor_msgs/tf2_sensor_msgs.h>

#include <tf/transform_listener.h>
#include <geometry_msgs/PointStamped.h>
#include <fstream>

#include <argo_track_msgs/Track_msg.h>
#include <track_array_msgs/TrackArray.h>
#include <argo_detection/DetectionArray.h>
#include <argo_detection/ArgoDetection.h>

// to sync the subscriber
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <boost/bind.hpp>

ros::Subscriber sub_detection;

// void draw_box(ros::Time lidar_timestamp){
//     visualization_msgs::Marker marker;
//     int i;
//     for(i=0; i<num; i++){
//         an temp = annotations.at(i);
//         marker.header.frame_id="/scan";
//         //marker.header.stamp = ros::Time();//to show every tag  (ros::Time::now()for latest tag)
//         marker.header.stamp = lidar_timestamp;
//         marker.ns = temp.track_class;
        
//         marker.action = visualization_msgs::Marker::ADD;
//         marker.lifetime = ros::Duration(de);
//         //bbox_marker.frame_locked = true;
//         marker.type = visualization_msgs::Marker::CUBE;
        
//         marker.id = i;
        
//         marker.pose.position.x = temp.center.x;
//         marker.pose.position.y = temp.center.y;
//         marker.pose.position.z = temp.center.z;

//         marker.pose.orientation.x = temp.rotation.x;
//         marker.pose.orientation.y = temp.rotation.y;
//         marker.pose.orientation.z = temp.rotation.z;
//         marker.pose.orientation.w = temp.rotation.w;
        
//         marker.scale.x = temp.length;
//         marker.scale.y = temp.width;
//         marker.scale.z = temp.height;
        
//         marker.color.b = 0.0f;
//         marker.color.g = 0.0f; //1.0f
//         marker.color.r = 0.0f;
// 	    marker.color.a = 0.7f;

//         // int k;
//         // for(k=1; k<category.size(); k++){
//         //     if( temp.track_class.compare(category.at(k)) == 0 ){
// 		//         //cout << category.at(k) <<endl;
//         //         int color = k % 3;
//         //         if ( color == 0 ){
//         //             marker.color.b = 0.0f + k*indent;
// 		//     //cout << "marker.color.b" << marker.color.b <<endl;
// 		// }
//         //         else if ( color == 1 ){
//         //             marker.color.g = 0.0f + k*indent;
// 		//     //cout << "marker.color.g" << marker.color.g <<endl;
// 		// }                
// 		// else{ 
//         //             marker.color.r = 0.0f + k*indent;
// 		//     //cout << "marker.color.r" << marker.color.r <<endl;
// 		// }
// 	    // }
                
//         // }
//         //seting color
//         float INDENT = (float)1/256;
//         if( temp.track_class == "VEHICLE")
//             marker.color.r = 1.0f;
//         else if( temp.track_class == "LARGE_VEHICLE")
//             marker.color.r = INDENT*205;
//         else if( temp.track_class == "BUS")
//             marker.color.r = INDENT*238;
//         else if( temp.track_class == "EMERGENCY_VEHICLE")
//             marker.color.r = INDENT*139;
//         else if( temp.track_class == "SCHOOL_BUS")
//             marker.color.r = INDENT*100;
//         else if( temp.track_class =="TRAILER")
//             marker.color.r = INDENT*70;
            
//         else if (temp.track_class == "PEDESTRIAN")
//             marker.color.g = 1.0f;

//         else if( temp.track_class == "BICYCLE")
//             marker.color.b = 1.0f;
//         else if( temp.track_class =="MOTORCYCLE")
//             marker.color.b = INDENT*238; 
//         else if( temp.track_class == "MOPED")
//             marker.color.b = INDENT*139;
        
//         else if( temp.track_class == "BICYCLIST"){
//             marker.color.r = INDENT*194;
//             marker.color.g = INDENT*228;
//             marker.color.b = INDENT*185;
//         }
//         else if( temp.track_class == "MOTORCYCLIST"){
//             marker.color.r = INDENT*127;
//             marker.color.g = INDENT*255;
//         }

//         else if( temp.track_class == "ON_ROAD_OBSTACLE"){
//             marker.color.r = INDENT*255;
//             marker.color.g = INDENT*77;
//             marker.color.b = INDENT*225;
//         }

//         else if( temp.track_class == "STROLLER"){
//             marker.color.r = INDENT*252;
//             marker.color.g = INDENT*185;
//             marker.color.b = INDENT*29;
//         }
//         else if( temp.track_class == "WHEELCHAIR"){
//             marker.color.r = INDENT*255;
//             marker.color.g = INDENT*165;
//         }
//         //OTHER_MOVER/ANIMAL
//         else if(temp.track_class == "OTHER_MOVER"){
//             marker.color.r = INDENT*160;
//             marker.color.g = INDENT*32;
//             marker.color.b = INDENT*240;
//         }
//         else{
//             marker.color.r = INDENT*218;
//             marker.color.g = INDENT*112;
//             marker.color.b = INDENT*214;
//         }

//         M_array.markers.push_back(marker);
//     }

//     if(M_array.markers.size() > max_size){
//         max_size = M_array.markers.size();
//     }

//     for (int a = i; a < max_size; a++)
//     {
//         marker.id = a;
//         marker.color.a = 0;
//         marker.pose.position.x = 0;
//         marker.pose.position.y = 0;
//         marker.pose.position.z = 0;
//         marker.scale.z = 0;
//         M_array.markers.push_back(marker);
//     }

//     pub_marker.publish(M_array);
    
// }

void callback(const argo_detection::DetectionArrayConstPtr &detections){
    std::cout << "We get detection at " << detections->header.stamp << endl;
}

int main(int argc,char** argv){
    ros::init(argc, argv, "viz_detection");
    ros::NodeHandle nh("~");

    sub_detection = nh.subscribe("/detection", 1000, &callback);
    while(ros::ok()){
        ros::spin();
    }
    return 0;
    

}