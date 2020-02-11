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


#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>


// #include "darknet_ros_msgs/BoundingBoxes.h"
// #include "darknet_ros_msgs/BoundingBox.h"


#include <opencv-3.3.1-dev/opencv2/core.hpp>
#include <opencv-3.3.1-dev/opencv2/highgui.hpp>
#include <opencv-3.3.1-dev/opencv2/imgproc.hpp>
#include <opencv-3.3.1-dev/opencv2/calib3d.hpp>
#include <opencv-3.3.1-dev/opencv2/opencv.hpp>


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

#include <signal.h>

ofstream gtFile,gt;
string gt_filename = "gt_original.csv";
string gt_modifiedID_filename = "gt_original_modified.csv";

tf::TransformListener *tf_listener;


ros::Subscriber track_id_sub;

using namespace std;

typedef struct an{
    geometry_msgs::Point center;
    geometry_msgs::Quaternion rotation;
    double length;
    double width;
    double height;
    string track_id;
    double time_s;
    string track_class;
    int occlusion;
 }an;

// vector <an> annotations;
vector <string> id_record;


void callback_gt(const track_array_msgs::TrackArray &t_a){
  track_array_msgs::TrackArray t = t_a;

  

  if(id_record.size() == 0){
    for (int i = 0; i<t.labels.size(); i++){
        argo_track_msgs::Track_msg msg = t.labels.at(i);

      geometry_msgs::PointStamped pt;
      geometry_msgs::PointStamped pt_transformed;
      pt.header = msg.header;
      pt.point.x =  msg.center.x;
      pt.point.y =  msg.center.y;
      pt.point.z =  msg.center.z;
      
      try{
          tf_listener->waitForTransform("/map","/scan",ros::Time(0),ros::Duration(5.0));//blocked process till get transform or 5 sec
          tf_listener->transformPoint("/map", pt, pt_transformed);
      }
      catch(tf::TransformException &ex) {
          ROS_WARN("%s", ex.what());
          // ros::Duration(1.0).sleep();
          continue;
      }

        id_record.push_back(msg.track_id);
        
        for (int k = 0; k<id_record.size(); k++){
            if( !(msg.track_id.compare(id_record.at(k))) )
                gt <<  msg.header.stamp << "," << k << "," << pt_transformed.point.x << "," << pt_transformed.point.y << "," << pt_transformed.point.z << std::endl; 
        }
            


        
        //local pose
        // id_record.push_back(msg.track_id);
        
        // for (int k = 0; k<id_record.size(); k++){
        //     if( !(msg.track_id.compare(id_record.at(k))) )
        //         gt <<  msg.header.stamp << "," << k << "," << msg.center.x << "," << msg.center.y << "," << msg.center.z << std::endl; 
        // }
            

        // gtFile << msg.header.stamp << "," << msg.track_id << "," << msg.center.x << "," << msg.center.y << "," << msg.center.z << std::endl; 
        
    }
  }
  else{
    for (int i = 0; i<t.labels.size(); i++){
        argo_track_msgs::Track_msg msg = t.labels.at(i);

      geometry_msgs::PointStamped pt;
      geometry_msgs::PointStamped pt_transformed;
      pt.header = msg.header;
      pt.point.x =  msg.center.x;
      pt.point.y =  msg.center.y;
      pt.point.z =  msg.center.z;
      
      try{
          tf_listener->waitForTransform("/map","/scan",ros::Time(0),ros::Duration(5.0));//blocked process till get transform or 5 sec
          tf_listener->transformPoint("/map", pt, pt_transformed);
      }
      catch(tf::TransformException &ex) {
          ROS_WARN("%s", ex.what());
          // ros::Duration(1.0).sleep();
          continue;
      }


        vector<string>::iterator find_id = find(id_record.begin(), id_record.end(),msg.track_id);
        if(find_id == id_record.end())
            id_record.push_back(msg.track_id);

        for (int k = 0; k<id_record.size(); k++){
            if( !(msg.track_id.compare(id_record.at(k))) )
                gt <<  msg.header.stamp << "," << k << "," << pt_transformed.point.x << "," << pt_transformed.point.y << "," << pt_transformed.point.z << std::endl; 
        }

       
        // local
        // vector<string>::iterator find_id = find(id_record.begin(), id_record.end(),msg.track_id);
        // if(find_id == id_record.end())
        //     id_record.push_back(msg.track_id);

        // for (int k = 0; k<id_record.size(); k++){
        //     if( !(msg.track_id.compare(id_record.at(k))) )
        //         gt <<  msg.header.stamp << "," << k << "," << msg.center.x << "," << msg.center.y << "," << msg.center.z << std::endl; 
        // }

        // gtFile << msg.header.stamp << "," << msg.track_id << "," << msg.center.x << "," << msg.center.y << "," << msg.center.z << std::endl; 
        
    }

  }
}

 
void MySigintHandler(int sig)
{
	//这里主要进行退出前的数据保存、内存清理、告知其他节点等工作
	ROS_INFO("shutting down!");
    gtFile.close();
    gt.close();
	ros::shutdown();
}


int main(int argc, char** argv){
    ros::init(argc, argv, "produce_gt");
    ros::NodeHandle nh;

    string out_dir = ros::package::getPath("lidar_track");

    
    track_id_sub = nh.subscribe("labels",1,&callback_gt);

    tf_listener = new tf::TransformListener() ;
    
    signal(SIGINT, MySigintHandler);

    gtFile.open(out_dir + "/output/" + gt_filename);
    gt.open(out_dir + "/output/" + gt_modifiedID_filename);
    id_record.clear();

    ros::Rate r(10);
    while(ros::ok()){
        ros::spinOnce();
        r.sleep();
    }


    return 0;

}
