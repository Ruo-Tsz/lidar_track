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

#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <jsk_recognition_msgs/BoundingBox.h>


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
#include <argo_detection/DetectionArray.h>
#include <argo_detection/ArgoDetection.h>

// to sync the subscriber
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <boost/bind.hpp>

ofstream outputFile;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2,
        												argo_detection::DetectionArray> GetSync;


using namespace std;
using namespace cv;
#define iteration 30 //plan segmentation # 30 for argo
#define sync_time 0.09 //10hz => 0.1sec, original 0.2s => one frame shift
string FRAME="/scan";  //output frmae_id

typedef pcl::PointXYZI PointT;
ros::Subscriber sub,label_sub;
ros::Publisher pub,pub_get,pub_colored;
ros::Publisher pub_voxel,pub_marker,pub_tra,pub_pt,pub_v,pub_pred, pub_self_v, pub_detection, pub_jsk_bbox;
ros::Publisher cluster_pub;

//get.pub image
image_transport::Subscriber img_sub;
image_transport::Publisher img_pub;

// tf::TransformListener listener;//declare in global would act like initialing a handler, it then before ros::init() and cause error 
tf::TransformListener *tf_listener; //Use a pointer to initialize in main instead

pcl::PointCloud<PointT>::Ptr cloud_pcl(new pcl::PointCloud<PointT>);
pcl::PointCloud<PointT>::Ptr cloud_pcl_whole(new pcl::PointCloud<PointT>);
pcl::PointCloud<PointT>::Ptr cloud_f(new pcl::PointCloud<PointT>);
pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);


vector<PointT> cens, cens_all;
vector<jsk_recognition_msgs::BoundingBox> jsk_bboxs;
visualization_msgs::MarkerArray m_s,l_s,current_m_a, tra_array, point_array, v_array, pred_point_array, self_v_array;
visualization_msgs::MarkerArray detection_array;
int max_size = 0;

float dt = 0.1f;//0.1f
float sigmaQ=0.01;//0.01
float sigmaR=0.1;//0.1
int id_count = 0; //the counter to calculate the current occupied track_id

////////////////////////kalman/////////////////////////
#define frame_lost 15
#define detect_thres 2.5 //l=4.5,w=1.8 (4.7,1.86 for nu)
#define bias 5.0 //5 //2.0
#define moving 1 //(10hz(0.1s) v>1m/s=60m/min=3.6km/hr = pedestrian)
#define invalid_v 30 // 108km/hr
#define show_tra 6
#define detection_thres 0.3f

bool get_label = false;

// KF init
int stateDim=10;// [x,y,z,v_x,v_y,v_z]  + w.h.l.theta
int measDim=7;// [x,y,z,l,w,h,theta]
int ctrlDim=0;// control input 0(acceleration=0,constant v model)
std::vector<pcl::PointCloud<PointT>::Ptr> cluster_vec;

std::vector<cv::KalmanFilter> KF;//(stateDim,measDim,ctrlDim,CV_32F);

cv::Mat state(stateDim,1,CV_32F);
cv::Mat_<float> measurement(3,1);//x.y.z pose as measurement

                        // measurement.setTo(Scalar(0));
bool firstFrame=true;

// //////type filter
typedef struct track{
  cv::KalmanFilter kf;
  geometry_msgs::Point pred_pose;
  geometry_msgs::Point pred_v;
  int lose_frame;
  int track_frame;
  string state;
  int cluster_idx;
  int uuid ;
  vector<geometry_msgs::Point> history;
  vector<float> S;
}track;


std::vector<track> filters;
ros::Time label_timestamp;

string dataset_,topic;
bool global_frame_, use_detection_, output_, debug_;


double get_yaw(const geometry_msgs::Quaternion quat){
  tf::Quaternion q(quat.x, quat.y, quat.z, quat.w);
  double roll, pitch, yaw;
  tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
  return yaw;
}


double correct_yaw(const geometry_msgs::Quaternion det_quat, double trk_yaw){
  double det_yaw = get_yaw(det_quat);
  double diff = det_yaw - trk_yaw;

  // correct from -2pi~2pi to -pi~pi
  if (diff > M_PI){
    det_yaw -= 2.0*M_PI;
  }
  else if (diff < (-1)*M_PI){
    det_yaw += 2.0*M_PI;
  }

  double diff_small = det_yaw - trk_yaw;
  // choose the smallest angle difference
  if (diff_small > M_PI/2.0){
    det_yaw -= M_PI;
  }
  else if (diff_small < M_PI/2.0){
    det_yaw += M_PI;
  }

  double diff_small_2 = det_yaw - trk_yaw;
  if (diff_small_2 > M_PI){
    det_yaw -= 2.0*M_PI;
  }
  else if (diff_small_2 < (-1)*M_PI){
    det_yaw += 2.0*M_PI;
  }

  return det_yaw;
}


// calculate euclidean distance of two points
double euclidean_distance(geometry_msgs::Point p1, geometry_msgs::Point p2)
{
  return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}

// double mahalanobis_distance(track trk, geometry_msgs::Point p2){
double mahalanobis_distance(track trk, jsk_recognition_msgs::BoundingBox p2){
  float t[measDim], det[measDim];
  t[0] = trk.pred_pose.x;
  t[1] = trk.pred_pose.y;
  t[2] = trk.pred_pose.z;
  t[3] = trk.kf.statePre.at<float>(3);
  t[4] = trk.kf.statePre.at<float>(4);
  t[5] = trk.kf.statePre.at<float>(5);
  t[6] = trk.kf.statePre.at<float>(6);

  det[0] = p2.pose.position.x;
  det[1] = p2.pose.position.y;
  det[2] = p2.pose.position.z;
  det[3] = p2.dimensions.x;
  det[4] = p2.dimensions.y;
  det[5] = p2.dimensions.z;
  det[6] = correct_yaw(p2.pose.orientation, trk.kf.statePre.at<float>(6));

  if (debug_){
    cout <<"After angle correction: " << det[6] - t[6] << endl;
  }
  cv::Mat tk = cv::Mat(measDim, 1, CV_32F, t);
  cv::Mat d = cv::Mat(measDim, 1, CV_32F, det);
  
  cv::KalmanFilter k = trk.kf;
  cv::Mat S = (k.measurementMatrix * k.errorCovPre) * k.measurementMatrix.t() + k.measurementNoiseCov;
  cv::Mat S_inv = S.inv();
  // cout << "The tracker " << trk.uuid << ", state " << std::setw(8) << trk.state << ", S_inv: " << S_inv.at<float>(0,0) <<", lose frame: "<<trk.lose_frame<< endl;
  double m_dist = cv::Mahalanobis(tk, d, S_inv);
  return sqrt( m_dist );
}


int find_matching(std::vector<double> dist_vec, std::vector<int> cens_matched){
  float now_min = std::numeric_limits<float>::max();
  int cluster_idx = -1;
  for (int i=0; i<dist_vec.size(); i++){
    if(dist_vec.at(i)<now_min && cens_matched.at(i) != 1){
      now_min = dist_vec.at(i);
      cluster_idx = i;
    }
  }
  cout<<"minIndex="<<cluster_idx<<endl;
  return cluster_idx;
}


pcl::PointCloud<PointT>::Ptr compute_c(pcl::PointCloud<PointT>::Ptr cloud_cluster,int j){
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud_cluster, centroid);
    pcl::PointXYZI c;
    c.x = centroid[0];
    c.y = centroid[1];
    c.z = centroid[2];
    c.intensity = j+100;
    cens.push_back(c);
    return cloud_cluster;
}



int new_track(const jsk_recognition_msgs::BoundingBox jsk_bbox, int idx){
    track tk;
    cv::KalmanFilter ka;
    ka.init(stateDim,measDim,ctrlDim,CV_32F);
    // ka.transitionMatrix = (Mat_<float>(6, 6) << 1,0,0,dt,0,0,
    //                                             0,1,0,0,dt,0,
    //                                             0,0,1,0,0,dt,
    //                                             0,0,0,1,0,0,
    //                                             0,0,0,0,1,0,
    //                                             0,0,0,0,0,1);
    // x,y,z,l,w,h,theat,vx,vy,vz
    ka.transitionMatrix = (Mat_<float>(stateDim, stateDim) << 1,0,0,0,0,0,0,dt,0,0,
                                                              0,1,0,0,0,0,0,0,dt,0,
                                                              0,0,1,0,0,0,0,0,0,dt,
                                                              0,0,0,1,0,0,0,0,0,0,
                                                              0,0,0,0,1,0,0,0,0,0,
                                                              0,0,0,0,0,1,0,0,0,0,
                                                              0,0,0,0,0,0,1,0,0,0,
                                                              0,0,0,0,0,0,0,1,0,0,
                                                              0,0,0,0,0,0,0,0,1,0,
                                                              0,0,0,0,0,0,0,0,0,1);
    cv::setIdentity(ka.measurementMatrix);
    cv::setIdentity(ka.processNoiseCov, Scalar::all(sigmaQ));
    cv::setIdentity(ka.measurementNoiseCov, cv::Scalar(sigmaR));
    ka.statePost.at<float>(0)= jsk_bbox.pose.position.x;
    ka.statePost.at<float>(1)= jsk_bbox.pose.position.y;
    ka.statePost.at<float>(2)= jsk_bbox.pose.position.z;
    ka.statePost.at<float>(3)= jsk_bbox.dimensions.x;// l
    ka.statePost.at<float>(4)= jsk_bbox.dimensions.y;// w
    ka.statePost.at<float>(5)= jsk_bbox.dimensions.z;// h
    ka.statePost.at<float>(6)= get_yaw(jsk_bbox.pose.orientation);// theta
    ka.statePost.at<float>(7)= 0;// initial v_x
    ka.statePost.at<float>(8)= 0;// initial v_y
    ka.statePost.at<float>(9)= 0;// initial v_z

    ka.errorCovPost = (Mat_<float>(stateDim, stateDim) << 1,0,0,0,0,0,0,0,0,0,
                                                          0,1,0,0,0,0,0,0,0,0,
                                                          0,0,1,0,0,0,0,0,0,0,
                                                          0,0,0,1,0,0,0,0,0,0,
                                                          0,0,0,0,1,0,0,0,0,0,
                                                          0,0,0,0,0,1,0,0,0,0,
                                                          0,0,0,0,0,0,1,0,0,0,
                                                          0,0,0,0,0,0,0,1000.0,0,0,
                                                          0,0,0,0,0,0,0,0,1000.0,0,
                                                          0,0,0,0,0,0,0,0,0,1000.0);

    

    tk.kf = ka;
    tk.state = "tracking";
    tk.lose_frame = 0;
    // tk.track_frame = 0;
    tk.track_frame = 1;
    tk.cluster_idx = idx;

    //
    geometry_msgs::Point pt_his;
    pt_his.x = jsk_bbox.pose.position.x;
    pt_his.y = jsk_bbox.pose.position.y;
    pt_his.z = jsk_bbox.pose.position.z;
    tk.history.push_back(pt_his);

    // tk.uuid = ++id_count;
    tk.uuid = id_count++;
    filters.push_back(tk);
    
    return tk.uuid;
}




pcl::PointCloud<PointT>::Ptr crop(pcl::PointCloud<PointT>::Ptr cloud_clusters){
  Eigen::Vector4f box_min,box_max;
  pcl::PointCloud<PointT>::Ptr cluster_box(new pcl::PointCloud<PointT> );
  box_min << -30,-30,-30,1;
  box_max << 30,30,30,1;//choose 60x60x60 cube (mask)
  pcl::CropBox<PointT> in_box;

  in_box.setInputCloud(cloud_clusters);
  in_box.setMin(box_min);
  in_box.setMax(box_max);
  in_box.filter(*cluster_box);

  // sensor_msgs::PointCloud2 cluster_cloud;
  // pcl::toROSMsg(*cluster_box, cluster_cloud);
  // cluster_cloud.header.frame_id = "map";
  // cluster_pub.publish(cluster_cloud);
  return cluster_box;
}


void get_bbox(){
  cout <<"BBOX"<<endl;
  get_label = false;
  
  // this way would cause exptorplation of tf in the past
  // tf::TransformListener listener; usually keep in the whole process like:member varible of class or global 
  


  for(int i=0; i<current_m_a.markers.size(); i++){
    visualization_msgs::Marker m = current_m_a.markers.at(i);
    // if a == 0 , is clear/fake marker, having no meaning 
    if ( m.color.a != 0 ){
      float x_m = m.pose.position.x;
      float y_m = m.pose.position.y;
      float z_m = m.pose.position.z;

      double or_x = m.pose.orientation.x;
      double or_y = m.pose.orientation.y;
      double or_z = m.pose.orientation.z;
      double or_w = m.pose.orientation.w;
      float scale_x = m.scale.x/2.0f;//l
      float scale_y = m.scale.y/2.0f;//w
      float scale_z = m.scale.z/2.0f;//h

      PointT bbox;
      bbox.x = x_m;
      bbox.y = y_m;
      bbox.z = z_m;
      cens.push_back(bbox);
    
      

      //transform gt box into map frame to tracking at map frame(motion segmention)
      // geometry_msgs::PointStamped pt;
      // geometry_msgs::PointStamped pt_transformed;
      // pt.header = current_m_a.markers[0].header;
      // pt.point.x = x_m;
      // pt.point.y = y_m;
      // pt.point.z = z_m;
      
      // try{
      //     tf_listener->waitForTransform("/map","/scan",ros::Time(0),ros::Duration(5.0));//blocked process till get transform or 5 sec
      //     tf_listener->transformPoint("/map", pt, pt_transformed);
      // }
      // catch(tf::TransformException &ex) {
      //     ROS_WARN("%s", ex.what());
      //     // ros::Duration(1.0).sleep();
      //     continue;

      // }
      // // using point/coordinate in map frame to tracking
      // PointT bbox;
      // bbox.x = pt_transformed.point.x;
      // bbox.y = pt_transformed.point.y;
      // bbox.z = pt_transformed.point.z;
      // cens.push_back(bbox);
    }


  }
}

void get_detection(const argo_detection::DetectionArray detections, tf::StampedTransform transform){
  argo_detection::DetectionArray objects = detections;
  cout << "In get_detection: " << detections.header.stamp << endl;;
  for(int i=0; i<objects.points.size(); i++){
    if ( objects.points.at(i).score < detection_thres ){
      continue;
    }
    geometry_msgs::PointStamped temp, transformed; 
    temp.point.x = objects.points.at(i).center.x;
    temp.point.y = objects.points.at(i).center.y;
    temp.point.z = objects.points.at(i).center.z;


    // transform detection from lidar frame to map frame
    tf::Point pt(temp.point.x, temp.point.y, temp.point.z);
    tf::Point ptTransform = transform * pt;

    // PointT bbox;
    // bbox.x = ptTransform.x();
    // bbox.y = ptTransform.y();
    // bbox.z = ptTransform.z();
    // cens.push_back(bbox);

    // transform local to global
    geometry_msgs::PoseStamped pose_global;
    tf::Transform pose_local;
    pose_local.setOrigin(tf::Vector3(objects.points.at(i).center.x, objects.points.at(i).center.y, objects.points.at(i).center.z));
    pose_local.setRotation(tf::Quaternion(objects.points.at(i).rotation.x, objects.points.at(i).rotation.y,\
                                         objects.points.at(i).rotation.z, objects.points.at(i).rotation.w));

    tf::poseTFToMsg( transform * pose_local, pose_global.pose);

    jsk_recognition_msgs::BoundingBox bbox;
    bbox.header.stamp = detections.header.stamp;
    bbox.header.frame_id = FRAME;
    bbox.pose = pose_global.pose;
    // bbox.pose.position.x = ptTransform.x();
    // bbox.pose.position.y = ptTransform.y();
    // bbox.pose.position.z = ptTransform.z();
    bbox.dimensions.x = objects.points.at(i).length;
    bbox.dimensions.y = objects.points.at(i).width;
    bbox.dimensions.z = objects.points.at(i).height;

    // !!!! quaternion也要轉
    // bbox.pose.orientation.x = objects.points.at(i).rotation.x;
    // bbox.pose.orientation.y = objects.points.at(i).rotation.y;
    // bbox.pose.orientation.z = objects.points.at(i).rotation.z;
    // bbox.pose.orientation.w = objects.points.at(i).rotation.w;
    jsk_bboxs.push_back(bbox);
    // bbox.label = ; uint32

    jsk_recognition_msgs::BoundingBoxArray jsk_bboxs_array;
    for (int i=0; i<jsk_bboxs.size(); i++){
      jsk_bboxs_array.header = jsk_bboxs.at(i).header;
      jsk_bboxs_array.boxes.push_back(jsk_bboxs.at(i));
    }
    pub_jsk_bbox.publish(jsk_bboxs_array);
  }

  return;
}


void ground_remove(const sensor_msgs::PointCloud2 out){
  pcl::fromROSMsg(out,*cloud_pcl_whole);  
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  // Create the segmentation object
  pcl::SACSegmentation<PointT> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (100);
  seg.setDistanceThreshold (0.02);//0.01 for bag1

  //Chose ROI to process
  //cloud_pcl = crop(cloud_pcl_whole);
  *cloud_pcl = *cloud_pcl_whole;

  cout<<"Ready to segmentation."<<endl;
  pcl::ExtractIndices<PointT> extract;
  pcl::console::TicToc ground_timer;
  ground_timer.tic();
  int i = 0;
  // for(i=0;i<iteration;i++){
  while( cloud_pcl->points.size() > 0.3*int(cloud_pcl_whole->points.size()) ){
  // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (cloud_pcl);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0)
    {
      std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
      break;
    }

    // Extract the inliers
    extract.setInputCloud (cloud_pcl);
    extract.setIndices (inliers);
    extract.setNegative (true);
    extract.filter (*cloud_f);
    cloud_pcl.swap (cloud_f);
    i++;
    //cout<<i<<endl;
  }
  cout<<"Ground removal takes "<<ground_timer.toc()/1000<<" secs.\n";
  cout<<"Before removal: " << cloud_pcl_whole->points.size()<<endl;
  cout<<"It runs "<<i<<" times.\n";
  cout<<"After removal: " << cloud_pcl->points.size() <<endl;
  cout<<"The ratio: "<<cloud_pcl->points.size()/cloud_pcl_whole->points.size() << endl;

  sensor_msgs::PointCloud2  output;
  pcl::toROSMsg(*cloud_pcl,output);
  output.header.frame_id = FRAME;
  pub.publish(output);
}


void euc_cluster(void){
    pcl::console::TicToc cluster_timer;
    cluster_timer.tic();
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud (cloud_filtered);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;

    ec.setClusterTolerance (0.45); //0.5->cars merged to one cluster
    ec.setMinClusterSize (40); //25 for clustering_1hz 6f bag 
    ec.setMaxClusterSize (400); //300 for train2,400 for train1(truck)
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud_filtered);
    ec.extract (cluster_indices);
    std::cout << "Cluster size: " << cluster_indices.size() << std::endl;
    
    cout<<"Cluster takes "<<cluster_timer.toc()/1000 << " secs.\n";

    int j = 50;

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_clusters (new pcl::PointCloud<pcl::PointXYZI>);
    
    cens_all.clear();
    cens.clear();
    cluster_vec.clear();

    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it) {
      
      // extract clusters and save as a single point cloud
      pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZI>);
      for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit) {
        cloud_filtered->points[*pit].intensity = j;
        cloud_cluster->points.push_back (cloud_filtered->points[*pit]); //*
      }
      cloud_cluster->width = cloud_cluster->points.size ();
      cloud_cluster->height = 1;
      cloud_cluster->is_dense = true;
      
      // compute cluster centroid and publish
      cloud_cluster = compute_c(cloud_cluster,j);
      cluster_vec.push_back(cloud_cluster);

      *cloud_clusters += *cloud_cluster;
      j+=2;

    }
    
    cout<< "cens_all ="<<cens_all.size()<<endl;
    cout<< "cens ="<<cens.size()<<endl;
    // cout << get_label << endl;

    sensor_msgs::PointCloud2 colored_pc;
    pcl::toROSMsg(*cloud_clusters, colored_pc);
    colored_pc.header.frame_id=FRAME;
    pub_colored.publish(colored_pc);
}


void filter_cluster(void){
  cout<<"Filtering the cluster.\n";
  get_label = false;

  float op_dist = std::numeric_limits<float>::max();
  std::vector <int> record_det(current_m_a.markers.size(),-1);


  for (int i = 0; i<cens_all.size(); i++){
    for (int j = 0; j<current_m_a.markers.size(); j++){
      visualization_msgs::Marker m = current_m_a.markers.at(j);
      // if a == 0 , is clear/fake marker, having no meaning 
      if ( m.color.a != 0 ){
        float x_m = m.pose.position.x;
        float y_m = m.pose.position.y;
        float z_m = m.pose.position.z;
    
        geometry_msgs::Point p_cen;//= geometry_msgs::Point(cens_all.at(i).x, cens_all.at(i).y, cens_all.at(i).z);
        float x = cens_all.at(i).x;
        float y = cens_all.at(i).y;
        float z = cens_all.at(i).z;
    
        double or_x = m.pose.orientation.x;
        double or_y = m.pose.orientation.y;
        double or_z = m.pose.orientation.z;
        double or_w = m.pose.orientation.w;
        float scale_x = m.scale.x/2.0f;//l
        float scale_y = m.scale.y/2.0f;//w
        float scale_z = m.scale.z/2.0f;//h

        
        
        ///(a) local = Tm^(-1) ^ tf^(-1) * global
        geometry_msgs::PointStamped pt;
        geometry_msgs::PointStamped pt_transformed;
        pt.header = current_m_a.markers[0].header;
        pt.point.x = x;
        pt.point.y = y;
        pt.point.z = z;

        try{
            tf_listener->waitForTransform("/map","/scan",ros::Time(0),ros::Duration(5.0));//blocked process till get transform or 5 sec
            tf_listener->transformPoint("/scan", pt, pt_transformed);
        }
        catch(tf::TransformException &ex) {
            ROS_WARN("%s", ex.what());
            // ros::Duration(1.0).sleep();
            continue;

        }
        
        Eigen::Quaternionf q = Eigen::Quaternionf(or_x, or_y, or_z, or_w);
        Eigen::Translation3f translation(x_m,y_m,z_m);
        Eigen::Affine3f transform = translation * q.toRotationMatrix();
        Eigen::Affine3f transform_inv = transform.inverse();
        Eigen::Vector3f v3f_a(pt_transformed.point.x, pt_transformed.point.y, pt_transformed.point.z);
        Eigen::Vector3f v3f_b = transform_inv*v3f_a;

        cout << "AFTER TRANS:\n" << v3f_b << endl; 
        ///
        
        /*
        /////(b) local = tf^(-1)* Tm^(-1) * global  failed
        Eigen::Quaternionf q = Eigen::Quaternionf(or_x, or_y, or_z, or_w);
        Eigen::Translation3f translation(x_m,y_m,z_m);
        Eigen::Affine3f transform = translation * q.toRotationMatrix();
        Eigen::Affine3f transform_inv = transform.inverse();
        Eigen::Vector3f v3f_a(x,y,z);
        Eigen::Vector3f v3f_b = transform_inv*v3f_a;

        cout << "Before TRANS:\n" << v3f_b << endl; 



        //v3f_b is in local origin in map frame
        
        geometry_msgs::PointStamped pt;
        geometry_msgs::PointStamped pt_transformed;
        pt.header = current_m_a.markers[0].header;
        pt.point.x = v3f_b[0];
        pt.point.y = v3f_b[1];
        pt.point.z = v3f_b[2];

        try{
            tf_listener->waitForTransform("/map","/scan",ros::Time(0),ros::Duration(5.0));//blocked process till get transform or 5 sec
            tf_listener->transformPoint("/scan", pt, pt_transformed);
        }
        catch(tf::TransformException &ex) {
            ROS_WARN("%s", ex.what());
            // ros::Duration(1.0).sleep();
            continue;

        }

        v3f_b[0] = pt_transformed.point.x;
        v3f_b[1] = pt_transformed.point.y;
        v3f_b[2] = pt_transformed.point.z;
        // cout << "AFTER TRANS:\n" << v3f_b << endl; 
        */
        
    
        if ( fabs(v3f_b[0]) <= scale_x && fabs(v3f_b[1]) <= scale_y ){
            if (record_det.at(j) == -1){
              record_det.at(j) = 0;
              cout << "HAVING "<<i<<"cluster"<<endl;
              cens.push_back(cens_all.at(i));
              break;
            }
        }
      }

    }
  }
}


void show_id(vector<int>obj_id){
    int k;
    visualization_msgs::Marker marker;
    for(k=0; k<jsk_bboxs.size(); k++){
    //   marker.header.frame_id="/nuscenes_lidar"; //child of pointcloud of tf map
      // marker.header.frame_id="/scan";
      // if ( !(global.compare("global")) )
      //   marker.header.frame_id="/map"; // coordinate transformed by tf to global map frame   //////////back TO MAP to get transformed global view in map!!
      // else
      //   marker.header.frame_id="/scan";
      marker.header.frame_id = FRAME;  
      marker.header.stamp = ros::Time();//to show every tag  (ros::Time::now()for latest tag)
      marker.action = visualization_msgs::Marker::ADD;
      marker.lifetime = ros::Duration(dt);
      marker.pose.orientation.w = 1.0;
      marker.id = k;
      marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;

      marker.scale.z = 1.5f;
      marker.color.b = 0.0f;//yellow
      marker.color.g = 0.9f;
      marker.color.r = 0.9f;
      marker.color.a = 1;

      geometry_msgs::Pose pose;
      // pose.position.x = cens[k].x;
      // pose.position.y = cens[k].y;
      // pose.position.z = cens[k].z+1.0f;
      pose.position = jsk_bboxs.at(k).pose.position;
      pose.position.z = jsk_bboxs.at(k).pose.position.z + 1.0f;
     
      stringstream ss;
      ss << obj_id.at(k);
      
      marker.text = ss.str();
      marker.pose = pose;
      m_s.markers.push_back(marker);
    }

    // if (m_s.markers.size() > max_size)
    //   max_size = m_s.markers.size();

    // for (int a = k; a < max_size; a++)
    // {
    //     marker.id = a;
    //     marker.color.a = 0;
    //     marker.pose.position.x = 0;
    //     marker.pose.position.y = 0;
    //     marker.pose.position.z = 0;
    //     marker.scale.z = 0;
    //     m_s.markers.push_back(marker);
    // }
    pub_marker.publish(m_s);
}


void show_trajectory(){
  int k=0;
  // visualization_msgs::Marker marker;
  for(k=0; k<filters.size(); k++){
    geometry_msgs::Point pred;
    pred.x = filters.at(k).pred_v.x;
    pred.y = filters.at(k).pred_v.y;
    pred.z = filters.at(k).pred_v.z;
    float velocity = sqrt(pred.x*pred.x + pred.y*pred.y + pred.z*pred.z);

    // if (velocity >= moving && velocity < invalid_v && !(filters.at(k).state.compare("tracking"))){  
    if (!(filters.at(k).state.compare("tracking"))){ 
      visualization_msgs::Marker marker, P;
      marker.header.frame_id = FRAME;  
      marker.header.stamp = ros::Time();//to show every tag  (ros::Time::now()for latest tag)
      marker.action = visualization_msgs::Marker::ADD;
      marker.lifetime = ros::Duration(dt);
      marker.pose.orientation.w = 1.0;
      marker.id = k;
      marker.type = visualization_msgs::Marker::LINE_STRIP;

      marker.scale.x = 0.1f;
      marker.color.g = 0.9f;
      marker.color.a = 1;


      P.header.frame_id = FRAME;
      P.header.stamp = ros::Time();
      P.action = visualization_msgs::Marker::ADD;
      P.lifetime = ros::Duration(dt);
      P.type = visualization_msgs::Marker::POINTS;
      P.id = k+1;
      P.scale.x = 0.3f;
      P.scale.y = 0.3f;
      P.color.r = 1.0f;
      P.color.a = 1;


      if (filters.at(k).history.size() < show_tra){
        for (int i=0; i<filters.at(k).history.size(); i++){
            geometry_msgs::Point pt = filters.at(k).history.at(i);
            marker.points.push_back(pt);
            P.points.push_back(pt);
        }
      }
      else{
        for (vector<geometry_msgs::Point>::const_reverse_iterator r_iter = filters.at(k).history.rbegin(); r_iter != filters.at(k).history.rbegin() + show_tra; ++r_iter){
            geometry_msgs::Point pt = *r_iter;
            marker.points.push_back(pt);
            P.points.push_back(pt);
        }
      }

      visualization_msgs::Marker Predict_p;
      Predict_p.header.frame_id = FRAME;
      Predict_p.header.stamp = ros::Time();
      Predict_p.action = visualization_msgs::Marker::ADD;
      Predict_p.lifetime = ros::Duration(dt);
      Predict_p.type = visualization_msgs::Marker::POINTS;
      Predict_p.id = k+2;
      Predict_p.scale.x = 0.4f;
      Predict_p.scale.y = 0.4f;
      Predict_p.color.r = (155.0f/255.0f);
      Predict_p.color.g = ( 99.0f/255.0f);
      Predict_p.color.b = (227.0f/255.0f);
      Predict_p.color.a = 1;
      geometry_msgs::Point pred = filters.at(k).pred_pose;
      Predict_p.points.push_back(pred);

    
      tra_array.markers.push_back(marker);
      point_array.markers.push_back(P);
      pred_point_array.markers.push_back(Predict_p);
    }
  }

  cout<<"We have "<<tra_array.markers.size()<< " moving objects."<<endl;

  pub_tra.publish(tra_array);
  pub_pt.publish(point_array);
  pub_pred.publish(pred_point_array);
}



void show_velocity(){
  for (int k=0; k<filters.size(); k++){

    geometry_msgs::Point pred;
    pred.x = filters.at(k).pred_v.x;
    pred.y = filters.at(k).pred_v.y;
    pred.z = filters.at(k).pred_v.z;
    float velocity = sqrt(pred.x*pred.x + pred.y*pred.y + pred.z*pred.z);

    // if ( velocity >= moving && velocity < invalid_v && !(filters.at(k).state.compare("tracking")) ){
    if ( !(filters.at(k).state.compare("tracking")) ){
      visualization_msgs::Marker arrow;
      arrow.header.frame_id = FRAME;
      arrow.header.stamp = ros::Time();
      arrow.lifetime = ros::Duration(dt);
      arrow.action = visualization_msgs::Marker::ADD;
      arrow.type = visualization_msgs::Marker::ARROW;
      arrow.id = k;
      
      geometry_msgs::Point tail, head;
      tail = filters.at(k).history.back();
      head.x = tail.x  + filters.at(k).pred_v.x;
      head.y = tail.y  + filters.at(k).pred_v.y;
      head.z = tail.z  + filters.at(k).pred_v.z;

      // cout << tail.x << "," << tail.y << endl;
      // cout << head.x << "," << head.y << endl;

      // arrow.points.at(0) = tail;
      // arrow.points.at(1) = head;

      arrow.points.push_back(tail);
      arrow.points.push_back(head);


      // cout << arrow.points.at(0).x << endl;
      // cout << arrow.points.at(1).x << endl;
      arrow.color.a = 1.0f;
      arrow.color.b = 0.7f;
      arrow.scale.x = 0.3f;
      arrow.scale.y = 0.6f;

      v_array.markers.push_back(arrow);
    }
      
    //output our self calculated velocity
    if( filters.at(k).history.size()>=2 && velocity >= moving && velocity < invalid_v && !(filters.at(k).state.compare("tracking"))){
      visualization_msgs::Marker self_arrow;

      self_arrow.header.frame_id = FRAME;
      self_arrow.header.stamp = ros::Time();
      self_arrow.lifetime = ros::Duration(dt);
      self_arrow.action = visualization_msgs::Marker::ADD;
      self_arrow.type = visualization_msgs::Marker::ARROW;
      self_arrow.id = k+1;
      
      geometry_msgs::Point tail, head, v;
      tail = filters.at(k).history.back();
      
      vector<geometry_msgs::Point>::iterator last = filters.at(k).history.end()-1;
      vector<geometry_msgs::Point>::iterator last_2 = filters.at(k).history.end()-2;
      v.x = ( (*last).x - (*last_2).x )/ dt;
      v.y = ( (*last).y - (*last_2).y )/ dt;
      v.z = ( (*last).z - (*last_2).z )/ dt;
      head.x = tail.x  + v.x;
      head.y = tail.y  + v.y;
      head.z = tail.z  + v.z;

  
      self_arrow.points.push_back(tail);
      self_arrow.points.push_back(head);


      // cout << arrow.points.at(0).x << endl;
      // cout << arrow.points.at(1).x << endl;
      self_arrow.color.a = 1.0f;
      self_arrow.color.r = ( 88.0f/255.0f);
      self_arrow.color.g = (183.0f/255.0f);
      self_arrow.color.b = (227.0f/255.0f);
      self_arrow.scale.x = 0.2f;
      self_arrow.scale.y = 0.4f;

      self_v_array.markers.push_back(self_arrow);
    }

  }

  pub_v.publish(v_array);
  pub_self_v.publish(self_v_array);
}


void KFT(ros::Time det_timestamp, bool use_mahalanobis)
{
  for(int i=0 ;i<filters.size() ;i++){
    geometry_msgs::Point pt, pt_v;
    cv::Mat pred;
    cv::KalmanFilter k = filters.at(i).kf;  
    pred = k.predict();
    pt.x = pred.at<float>(0);
    pt.y = pred.at<float>(1);
    pt.z = pred.at<float>(2);
    pt_v.x = pred.at<float>(7);
    pt_v.y = pred.at<float>(8);
    pt_v.z = pred.at<float>(9);
    filters.at(i).pred_pose = pt;
    filters.at(i).pred_v = pt_v;
  }

  std::vector<geometry_msgs::Point> clusterCenters;
  
  int i=0;
  for(i; i<jsk_bboxs.size(); i++){
    geometry_msgs::Point pt;
    pt = jsk_bboxs.at(i).pose.position;
    clusterCenters.push_back(pt);
  }

  i=0;

  //construct dist matrix (mxn): m tracks, n clusters.
  std::vector<std::vector<double> > distMat; //float

  for(std::vector<track>::const_iterator it = filters.begin (); it != filters.end (); ++it)
  {
      std::vector<double> distVec; //float
      if (!use_mahalanobis){
        for(int n=0;n<jsk_bboxs.size();n++)
        {
          distVec.push_back(euclidean_distance((*it).pred_pose,clusterCenters[n]));
        }
      }
      else{
        cv::KalmanFilter k = (*it).kf;
        cv::Mat S = (k.measurementMatrix * k.errorCovPre) * k.measurementMatrix.t() + k.measurementNoiseCov;
        cv::Mat S_inv = S.inv();
        if( debug_ ){
          cout << "The tracker " << std::setw(3) << (*it).uuid << ", state " << std::setw(8) << (*it).state << ", S_inv: " << std::fixed << std::setprecision(5) << S_inv.at<float>(0,0) <<", lose frame: "<<(*it).lose_frame<< endl;
        }
        for(int n=0;n<jsk_bboxs.size();n++)
        {
          // distVec.push_back( mahalanobis_distance( (*it), clusterCenters[n] ));
          distVec.push_back( mahalanobis_distance( (*it), jsk_bboxs.at(n) ));
        }
      }
      
      distMat.push_back(distVec);
  }

  
  //-----------------modified hun, find the nearest cluster first, then optimize by Hungarian

  // std::vector<int> cens_matched(cens.size(),-1); 
  // std::vector<int> track_matched_id(filters.size(), -1);
  // std::vector<int> track_matched_idx, cens_matched_idx;
  // int k=0, matched=0;
  // int cluster_idx = -1;
  // std::pair<int,int> minIndex;
  // //every track to find the min value

  // for(k=0; k<filters.size(); k++){
  //   track *current_track = &(filters.at(k));
  //   std::vector<double> dist_vec = distMat.at(k);
  //   std::vector<double> matched_distVec;

  //   cluster_idx = find_matching(dist_vec, cens_matched);

  //   if (cluster_idx != -1){
  //     if( fabs(cens.at(cluster_idx).x - current_track->history.back().x) < fabs(current_track->pred_v.x) * dt + bias && \
  //         fabs(cens.at(cluster_idx).y - current_track->history.back().y) < fabs(current_track->pred_v.y) * dt + bias ){//bias as gating function to filter the impossible matched detection 
        
  //         cens_matched[cluster_idx] = 1;
  //         track_matched_id[k] = cluster_idx;
  //         matched++;
  //         track_matched_idx.push_back(k);
  //         cens_matched_idx.push_back(cluster_idx);
  //         current_track->state = "tracking";
  //         std::cout<<"track "<< current_track->uuid << " tracking. track: "<<k<<" ,clus: "<<cluster_idx << std::endl;
  //         std::cout<<"The distance is "<< distMat.at(k).at(cluster_idx) << std::endl;
  //       }
  //     else{
  //       std::cout<<"track "<< current_track->uuid << " losting, not qualified.\n";
  //       std::cout<<"The distance is "<< distMat.at(k).at(cluster_idx) << std::endl;
  //       current_track->state = "lost";
  //     }
  //   }
  //   else{
  //     std::cout<<"track "<< current_track->uuid << " losting, already matched.\n";
  //     current_track->state = "lost";
  //   }
  // }

  // for (int i=0; i<track_matched_idx.size(); i++)
  //     cout<< track_matched_idx.at(i) <<", ";
  // cout<<endl;
  // for (int ii=0; ii<cens_matched_idx.size(); ii++)
  //     cout<< cens_matched_idx.at(ii) <<", ";
  // cout<<endl;

  // // initiate matched_dist_matrix to Hungarian, get optimal global assignment
  // std::vector<std::vector<double> > matched_distMat;
  // for(k=0; k<track_matched_idx.size(); k++){
  //   std::vector<double> matched_distVec;
  //   for(int jj=0; jj<cens_matched_idx.size(); jj++){
      
  //     matched_distVec.push_back( euclidean_distance( filters.at(track_matched_idx.at(k)).pred_pose, clusterCenters.at(cens_matched_idx.at(jj)) ) );
    
  //   }
  //   matched_distMat.push_back(matched_distVec);
  // }


  // // for (int i=0; i<matched_distMat.size(); i++){
  // //   vector<double> dist = matched_distMat.at(i);
  // //   for(int j=0; j<matched_distMat.at(0).size(); j++){
  // //     cout<<dist.at(j)<<", ";
  // //   }
  // //   cout<<endl;
  // // }
  
  // // do Hungarian
  // HungarianAlgorithm HungAlgo;
  // vector<int> assignment;
  // std::vector<int> obj_id(cens.size(),-1); 

  // double cost = HungAlgo.Solve(matched_distMat, assignment);

  // for (unsigned int x = 0; x < matched_distMat.size(); x++)
  //   std::cout << x << "," << assignment[x] << "\t";

  // std::cout << "\ncost: " << cost << std::endl;

  // for (k=0; k<track_matched_idx.size(); k++){
  //   int clus_idx = assignment.at(k);
  //   int track_idx = track_matched_idx.at(k);
  //   filters.at(track_idx).cluster_idx = cens_matched_idx.at(clus_idx);
  //   obj_id[cens_matched_idx.at(clus_idx)] = filters.at(track_idx).uuid;
  // }

  
  
  //-----------------
  
  // ///////////////////original whole Hungarian
  // hungarian method to optimize(minimize) the dist matrix
  HungarianAlgorithm HungAlgo;
  vector<int> assignment;

  double cost = HungAlgo.Solve(distMat, assignment);

  // for (unsigned int x = 0; x < distMat.size(); x++)
  //   std::cout << x << "," << assignment[x] << "\t";
  // std::cout << "\ncost: " << cost << std::endl;


  std::vector<int> obj_id(jsk_bboxs.size(),-1); 
  int k=0;
  for(k=0; k<filters.size(); k++){
    std::vector<double> dist_vec = distMat.at(k); //float
    track *current_track = &(filters.at(k));

    geometry_msgs::Point pred_v = filters.at(k).pred_v;
    float velocity = sqrt(pred_v.x * pred_v.x + pred_v.y * pred_v.y);
    
    //-1 for non matched tracks, gating function : use velocity to filter out impossible matching
    if ( assignment[k] != -1 ){
      int clus_idx = assignment[k];
      if (fabs( jsk_bboxs.at(clus_idx).pose.position.x - current_track->history.back().x ) < fabs(current_track->pred_v.x) * dt + bias && \
          fabs( jsk_bboxs.at(clus_idx).pose.position.y - current_track->history.back().y ) < fabs(current_track->pred_v.y) * dt + bias){
        
        obj_id.at(clus_idx) = current_track->uuid;
        current_track->cluster_idx = clus_idx;
        current_track->state = "tracking";
        // if (debug_){
          cout << "The track " << current_track->uuid << " dist is " << distMat[k][clus_idx] << endl;
      //   }
      }
      else{
        // if(debug_){
          cout << "FILTERED OUT！！" << " The track " << current_track->uuid << " dist is " << distMat[k][clus_idx] << endl;
        // }
        current_track->state = "lost";
      }  
    }
    else // tracks missed(not correspondance build)
    {
      current_track->state= "lost";
    }
    

    
  }  
  ///////////////------------------------------original hungarian

  // cope with existing/old track (not new_track yet), deal with lost tracks and let it remain const velocity -> update pred_pose to meas correct
  for(std::vector<track>::iterator pit = filters.begin (); pit != filters.end ();){
      if( !(*pit).state.compare("lost")  ){//true for 0
          (*pit).lose_frame += 1;
          //record the pred as tracking trajectory
          geometry_msgs::Point pt_his = (*pit).pred_pose;
          (*pit).history.push_back(pt_his);
      }

      if( !(*pit).state.compare("tracking")  ){//true for 0
          (*pit).track_frame += 1;
          (*pit).lose_frame = 0;

          //record the tracking trajectory
          geometry_msgs::Point pt_his;
          pt_his = jsk_bboxs.at((*pit).cluster_idx).pose.position;
          // pt_his.x = cens.at((*pit).cluster_idx).x;
          // pt_his.y = cens.at((*pit).cluster_idx).y;
          // pt_his.z = cens.at((*pit).cluster_idx).z;
          (*pit).history.push_back(pt_his);

      }    
      
      if(debug_){
        if (!(*pit).state.compare("tracking"))
           cout<< std::setw(3) << (*pit).uuid<<" tracker is \033[1;34m"<< std::setw(8) <<(*pit).state<< std::setw(9) << "\033[0m, track: "<<(*pit).track_frame << ", to cluster_idx "<< (*pit).cluster_idx <<endl;
        else if (!(*pit).state.compare("lost"))
            cout<< std::setw(3) <<(*pit).uuid<<" tracker is \033[1;34m"<< std::setw(8) <<(*pit).state<< std::setw(9) << "\033[0m, lost: "<<(*pit).lose_frame<<endl;
        else
            cout<<"\033[1;31mUndefined state for tracked "<<k<<"\033[0m"<<endl;
      }

      // if we lose track consecutively ''frame_lost'' frames, remove track
      if ( (*pit).lose_frame == frame_lost )
          //remove track from filters
          pit = filters.erase(pit);
      else
          pit ++;
          
  }


  /*
  for (int j=0; j<filters.size(); j++){
    // if(filters.at(j).uuid == 9){
    geometry_msgs::Point pred;
    pred.x = filters.at(j).pred_v.x;
    pred.y = filters.at(j).pred_v.y;
    pred.z = filters.at(j).pred_v.z;
    float velocity = sqrt(pred.x*pred.x + pred.y*pred.y + pred.z*pred.z);

    // if(velocity >= moving && velocity <= 5 && !(filters.at(j).state.compare("tracking")) ){
    if(velocity >= moving && velocity <= invalid_v && !(filters.at(j).state.compare("tracking")) ){
      cout<<"The state of "<<filters.at(j).uuid<<" filters is \033[1;34m"<<filters.at(j).state<<"\033[0m,to cluster_idx "<< filters.at(j).cluster_idx <<" track: "<<filters.at(j).track_frame<<endl;
      // cout <<"The tra of "<<filters.at(j).uuid<<" is:"<<endl;
      // for (int i=0; i<filters.at(j).history.size(); i++){
      //   geometry_msgs::Point pt = filters.at(j).history.at(i);
      //   cout<<pt.x<<","<<pt.y<<","<<pt.z<<endl;
      // }
      // cout<<"----------------------------------"<<endl;
    }
  }
  */
  
  ///////////////////////////////////////////////////estimate(update old/existing tracks)
  int num = filters.size();
  float meas[num][measDim];
  i = 0;
  for(std::vector<track>::const_iterator it = filters.begin(); it != filters.end(); ++it){
      if ( (*it).state == "tracking" ){
          // PointT pt = cens[(*it).cluster_idx];
          // meas[i][0] = pt.x;
          // meas[i][1] = pt.y;
          // meas[i][2] = pt.z;
          jsk_recognition_msgs::BoundingBox bbox = jsk_bboxs.at((*it).cluster_idx);
          meas[i][0] = bbox.pose.position.x;
          meas[i][1] = bbox.pose.position.y;
          meas[i][2] = bbox.pose.position.z;
          meas[i][3] = bbox.dimensions.x;
          meas[i][4] = bbox.dimensions.y;
          meas[i][5] = bbox.dimensions.z;
          meas[i][6] = get_yaw(bbox.pose.orientation);
          
      }
      else if ( (*it).state == "lost" ){
          // meas[i][0] = (*it).pred_pose.x; //+ (*it).pred_v.x;
          // meas[i][1] = (*it).pred_pose.y;//+ (*it).pred_v.y;
          // meas[i][2] = (*it).pred_pose.z; //+ (*it).pred_v.z;
      }
      else{
          std::cout<<"Some tracks state not defined to tracking/lost."<<std::endl;
      }
      i++;
  }

  ROS_INFO("Measurement update");
  cv::Mat measMat[num];
  for(int i=0;i<num;i++){
      measMat[i]=cv::Mat(measDim,1,CV_32F,meas[i]);
  }

  // The update phase 
  cv::Mat estimated[num];
  i = 0;
  for(std::vector<track>::iterator it = filters.begin(); it != filters.end(); ++it){
      if ((*it).state == "tracking")
        estimated[i] = (*it).kf.correct(measMat[i]); 
      i++;
  }


  //initiate new tracks for unmatched cluster obj.at(cluster_idx) == -1
  for (i=0; i<jsk_bboxs.size(); i++){
    if(obj_id.at(i) == -1){
      int track_uuid = new_track(jsk_bboxs.at(i),i);
      obj_id.at(i) = track_uuid;
    }
  }


  // obj_id now is the traking result of all cens, done associate. if remain -1 => false positive(alarm)
  //////output result
  if( output_ ){
    for (int i=0; i<obj_id.size(); i++){
      outputFile << det_timestamp << "," << obj_id.at(i) << "," << cens.at(i).x << "," << cens.at(i).y << "," << cens.at(i).z << endl; 
      // outputFile << label_timestamp << "," << obj_id.at(i) << "," << cens.at(i).x << "," << cens.at(i).y << "," << cens.at(i).z << endl;//in get_bbox, we don't care lidar point cloud, just get marker and do
    }
  }

  ROS_INFO("We now have %d tracks.", (int)filters.size());

  m_s.markers.resize(jsk_bboxs.size());
  m_s.markers.clear();
  tra_array.markers.clear();
  point_array.markers.clear();
  v_array.markers.clear();
  pred_point_array.markers.clear();
  self_v_array.markers.clear();
  
  show_id(obj_id);
  show_trajectory();
  show_velocity();
  return;
}


void draw_box(const argo_detection::DetectionArray& detection){
  visualization_msgs::Marker marker;
  detection_array.markers.clear();
    
  for (int i=0; i<detection.points.size(); i++){
    argo_detection::ArgoDetection single = detection.points.at(i);
    if ( single.score < detection_thres ){
      continue;
    }
    // cout << single.label_class << endl;
    marker.header.frame_id = topic;
    marker.header.stamp = detection.header.stamp;
    marker.ns = "detection_marker";
    marker.action = visualization_msgs::Marker::ADD;
    marker.lifetime = ros::Duration(dt);
    marker.type = visualization_msgs::Marker::CUBE;
    marker.id = i;

    marker.pose.position = single.center;
    marker.pose.orientation = single.rotation;
    marker.scale.x = single.length;
    marker.scale.y = single.width;
    marker.scale.z = single.height;
    marker.color.a = 0.7f;
    marker.color.b = 0.0f;
    marker.color.g = 0.0f; //1.0f
    marker.color.r = 0.0f;

    float INDENT = (float)1/256;
    if( single.label_class == "VEHICLE")
        marker.color.r = 1.0f;
    else if( single.label_class == "LARGE_VEHICLE")
        marker.color.r = INDENT*205;
    else if( single.label_class == "BUS")
        marker.color.r = INDENT*238;
    else if( single.label_class == "EMERGENCY_VEHICLE")
        marker.color.r = INDENT*139;
    else if( single.label_class == "SCHOOL_BUS")
        marker.color.r = INDENT*100;
    else if( single.label_class =="TRAILER")
        marker.color.r = INDENT*70;
        
    else if (single.label_class == "PEDESTRIAN")
        marker.color.g = 1.0f;

    else if( single.label_class == "BICYCLE")
        marker.color.b = 1.0f;
    else if( single.label_class =="MOTORCYCLE")
        marker.color.b = INDENT*238; 
    else if( single.label_class == "MOPED")
        marker.color.b = INDENT*139;
    
    else if( single.label_class == "BICYCLIST"){
        marker.color.r = INDENT*194;
        marker.color.g = INDENT*228;
        marker.color.b = INDENT*185;
    }
    else if( single.label_class == "MOTORCYCLIST"){
        marker.color.r = INDENT*127;
        marker.color.g = INDENT*255;
    }

    else if( single.label_class == "ON_ROAD_OBSTACLE"){
        marker.color.r = INDENT*255;
        marker.color.g = INDENT*77;
        marker.color.b = INDENT*225;
    }

    else if( single.label_class == "STROLLER"){
        marker.color.r = INDENT*252;
        marker.color.g = INDENT*185;
        marker.color.b = INDENT*29;
    }
    else if( single.label_class == "WHEELCHAIR"){
        marker.color.r = INDENT*255;
        marker.color.g = INDENT*165;
    }
    //OTHER_MOVER/ANIMAL
    else if(single.label_class == "OTHER_MOVER"){
        marker.color.r = INDENT*160;
        marker.color.g = INDENT*32;
        marker.color.b = INDENT*240;
    }
    else{
        marker.color.r = INDENT*218;
        marker.color.g = INDENT*112;
        marker.color.b = INDENT*214;
    }

    detection_array.markers.push_back(marker);

  }
  pub_detection.publish(detection_array);
}


// void callback(const sensor_msgs::PointCloud2 &msg){
void callback(const sensor_msgs::PointCloud2ConstPtr& cloud, const argo_detection::DetectionArrayConstPtr& detection){
  sensor_msgs::PointCloud2 msg = *cloud;
  ros::Time lidar_timestamp = msg.header.stamp;
  ros::Time det_timestamp = detection->header.stamp;
  float time_dif;
  cout<<"\033[1;33mWe now @ "<<lidar_timestamp<<"\033[0m"<<endl;
  cout<<"Detection stamp " << det_timestamp << endl;



  // 原本的get  會產生lidar永遠比lable快0.1秒（1禎）,一直lose, tracking糟
  // 最原始sync_time = 0.2 => 非常即時的shift一個frame()
  // if ( !get_label ){
  //     cout<<"Not detect. @"<<lidar_timestamp<<endl;
  //     return;
  // }
  // // For gt_label 先關掉 直接label來就做
  // else{
  //     time_dif = lidar_timestamp.toSec() - label_timestamp.toSec();
  //     if (fabs(time_dif) >= sync_time){
  //         cout<<"Not detect. @ lidar: "<<lidar_timestamp<<", label: "<<label_timestamp<<endl;
  //         cout<<"Lidar to sec " << lidar_timestamp.sec << ", "<<label_timestamp.nsec<<endl;
  //         long double time = ( lidar_timestamp.sec%10000 + lidar_timestamp.nsec/1000000000.0 )*10.0 ;
  //         // long double time =  (lidar_timestamp.nsec/100000000.0) + lidar_timestamp.sec * 10;
  //         cout<<fixed<<setprecision(9)<<time<< endl;
  //         long int shift = time ;
  //         cout<< shift << endl;
  //         // cout<< ((lidar_timestamp.nsec) / 1000000000.0) << endl;
  //         // cout<< ( label_timstamp * 10 )%1 << endl;
  //         return;
  //     }
  // }


  sensor_msgs::PointCloud2 out;
  sensor_msgs::PointCloud out_2,out_transformed_2;
  out = msg;

  //transform point cloud to map frame by tf
  tf::StampedTransform transform;
  if ( global_frame_ ){
    sensor_msgs::convertPointCloud2ToPointCloud(out, out_2);
    try {
        // tf_listener->waitForTransform("/map", "/scan", ros::Time(0), ros::Duration(5.0));
        tf_listener->waitForTransform("/map", topic, lidar_timestamp, ros::Duration(1.0));
        tf_listener->lookupTransform("/map", topic, lidar_timestamp, transform);
        // pcl_ros::transformPointCloud("/map", *cloud_pcl_whole, *cloud_pcl, (*tf_listener)); //pcl_ros’ has not been declared
        tf_listener->transformPointCloud("/map", out_2, out_transformed_2);
        // tf_listener->transformPointCloud("/map", )
    }
    catch(tf::TransformException &ex)
    {
        ROS_WARN("%s", ex.what());
        return;
    }  
    sensor_msgs::convertPointCloudToPointCloud2(out_transformed_2, out);
  }

  ///test
  out.header.frame_id="/map";
  pub_get.publish(out);
  cout<<"I get the car. at "<<lidar_timestamp<<endl;
  ///test
  
  if (!use_detection_){
    // 改用ground_filter進來的點雲 /no_ground
    // GROUND
    ground_remove(out);

    ////////////////////no downsampling
    pcl::VoxelGrid<PointT> sor;
    sor.setInputCloud (cloud_pcl);
    sor.setLeafSize (0.25f, 0.25f, 0.25f); //0.25 for bag1; 0.1 for 3 //vedio 0.35 not many pedestrain,0.25better
    sor.filter (*cloud_filtered);
    sensor_msgs::PointCloud2 cloud_filtered_sensor;
    pcl::toROSMsg(*cloud_filtered,cloud_filtered_sensor);
    cloud_filtered_sensor.header.frame_id = FRAME;
    pub_voxel.publish(cloud_filtered_sensor);

    //cluster
    euc_cluster();
 
  }
  /////////////////////////////done cluster
  else{
    cens.clear();
    jsk_bboxs.clear();
    get_detection(*detection, transform);
  }

  
  
  //(1)
  /////////////////////////////////////////////remove above preprocessing point cloud and from filter_cluster just get bbox to see tracking performance (1)
  /*
  cens.clear(); //(2)
  if ( get_label ){
    // float time_dif = lidar_timestamp.toSec() - label_timestamp.toSec();
    if (fabs(time_dif) < sync_time){
      cout <<" Differ time"<<fabs(time_dif) << endl;
      // filter_cluster(); //(1)
      get_bbox(); //(2)
    }
    else{
      cout<<"Not detect. yo\n";
      return;
    }
  }  
  else{
    cout<<"Not detect.\n";
    return;
  }
  */

  if( firstFrame ){
    int current_id = jsk_bboxs.size();
    for (int i=0; i<current_id; i++){
      int uuid = new_track(jsk_bboxs.at(i), i);
    }
    cout << "We have " << jsk_bboxs.size() << " detections" << endl;
    cout<<"Initiate "<<filters.size()<<" tracks."<<endl;

    m_s.markers.clear();
    std::vector<int> obj_id(jsk_bboxs.size(),-1);
    for(int i =0; i<jsk_bboxs.size(); i++){
      obj_id.at(i) = i;
    }
    show_id(obj_id);
    
    firstFrame=false;
    return;
  }

  KFT(det_timestamp, true);
  draw_box(*detection);
                 
  /* (1)
  cloud_clusters = crop(cloud_clusters);
  sensor_msgs::PointCloud2 cluster_cloud;
  pcl::toROSMsg(*cloud_clusters, cluster_cloud);
//   cluster_cloud.header.frame_id = "/nuscenes_lidar";
  cluster_cloud.header.frame_id = "/scan";
  cluster_pub.publish(cluster_cloud);
  */ 

  return;

}


void callback_label(const visualization_msgs::MarkerArray &m_a){
  current_m_a = m_a;
  label_timestamp = m_a.markers[0].header.stamp;
  get_label = true;
  cout<<"Get new label, at "<<label_timestamp<<endl;
  return;
}

/*
void param_parser(string out_dir){

  nh.getParam("dataset", dataset);
  nh.getParam("param", param);
  nh.getParam("global",global);  
  
  if (!(dataset.compare("nuscene"))){
    topic = "/nuscenes_lidar";
    cout<<"On nuscene dataset, subscribe to "<< topic <<endl;
  }
  else if (!(dataset.compare("argo"))){
    topic = "/scan";
    cout<<"On argo dataset, subscribe to "<< topic <<endl;
  }
  else{
    cout<<"Please provide the dataset you use as _dataset:= DATASET.\n";
    return;
  }


  if( !(param.compare("output")) ){
    string filename = "trackOutput_test.csv";
    outputFile.open(out_dir + "/output/" + filename);
    cout << "Output file " << filename <<endl;
  }
  else
    cout << "No output" << endl;


  if( !(global.compare("global")) ){
    FRAME = "/map";
  }
  else{
    FRAME = "/scan";
  }

  cout<<"We now at "<< FRAME << " frame." << endl;
  return;
}
*/

int main(int argc, char** argv){
  ros::init(argc,argv,"tracking");
  ros::NodeHandle nh("~");
  string out_dir = ros::package::getPath("lidar_track");

  // rosrun lidar_track kal_update _dataset:=argo _output:= _global_frame:=true _use_detection:=true
  nh.param<string>("dataset", dataset_, "argo");
  nh.param<bool>("output", output_, false);
  nh.param<bool>("global_frame", global_frame_, true);
  nh.param<bool>("use_detection", use_detection_, true);
  nh.param<bool>("debug", debug_, false);

  //用nh.param("globel", global, defaultvalue);
  //在callback裡面取得msg.header.frame_id當成inputid, 這個FRAME為outputid, 用這個方式轉transform

  if (!(dataset_.compare("nuscene"))){
    topic = "/nuscenes_lidar";
    ROS_INFO("On nuscene dataset, subscribe to %s", topic.c_str());
  }
  else if (!(dataset_.compare("argo"))){
    topic = "/scan";
    // topic = "/point_cloud";
    ROS_INFO("On argo dataset, subscribe to %s", topic.c_str());
  }
  else{
    ROS_INFO("Please provide the dataset you use as _dataset:= DATASET.");
    return -1;
  }


  // if( !(param.compare("output")) ){
  if ( output_ ){
    string filename = "detection_6f.csv";
    outputFile.open(out_dir + "/output/" + filename);
    ROS_INFO("Output file : %s", filename.c_str());
    // cout << "Output file " << filename <<endl;
  }
  else
    ROS_INFO("No output recorded.");

  if( global_frame_ ){
    FRAME = "/map";
  }
  else{
    if (!(dataset_.compare("nuscene")))
      FRAME = "/nuscenes_lidar";
    else
      FRAME = "/scan"; 
  }

  ROS_INFO("We now at %s frame", FRAME.c_str());
  ROS_INFO("Using detection : %s", (use_detection_ ? "true" : "false") );

 
  
  // sub = nh.subscribe("points_raw",1000,&callback);
  // sub = nh.subscribe("/nuscenes_lidar",1,&callback);
  // label_sub = nh.subscribe("lidar_label",1,&callback_label);
  // sub = nh.subscribe("/scan",1,&callback);
  // label_sub = nh.subscribe("/anno_marker",1,&callback_label);

  // sub = nh.subscribe( topic,1,&callback);

  // sub = nh.subscribe( "/no_ground",1,&callback);
  message_filters::Subscriber<sensor_msgs::PointCloud2> sub(nh, topic, 100);
  message_filters::Subscriber<argo_detection::DetectionArray> det_sub(nh, "/detection", 100);
  // message_filters::Subscriber<argo_detection::DetectionArray> det_sub(nh, "/detection/lidar_objects", 500);
  
  // message_filters::TimeSynchronizer<sensor_msgs::PointCloud2, argo_detection::DetectionArray> sync(sub, det_sub, 100);
  // sync.registerCallback(boost::bind(&callback, _1, _2));
  message_filters::Synchronizer<GetSync>* sync_filter;
  sync_filter = new message_filters::Synchronizer<GetSync>(GetSync(3), sub, det_sub);
  sync_filter->registerCallback(boost::bind(&callback, _1, _2));
  
  pub_get = nh.advertise<sensor_msgs::PointCloud2>("original_pc", 1000);
  pub = nh.advertise<sensor_msgs::PointCloud2>("no_ground", 1000);
  pub_voxel = nh.advertise<sensor_msgs::PointCloud2>("voxel_pc", 1000);
  pub_colored = nh.advertise<sensor_msgs::PointCloud2>("colored_pc",1000);

  cluster_pub = nh.advertise<sensor_msgs::PointCloud2>("cluster_pc", 1);
  pub_marker = nh.advertise<visualization_msgs::MarkerArray>("id_marker", 1);
  pub_tra = nh.advertise<visualization_msgs::MarkerArray>("trajectory_marker",1);
  pub_pt = nh.advertise<visualization_msgs::MarkerArray>("pt_marker", 1);
  pub_v = nh.advertise<visualization_msgs::MarkerArray>("velocity_marker", 1);
  pub_pred = nh.advertise<visualization_msgs::MarkerArray>("pred_pose_marker",1);
  pub_self_v = nh.advertise<visualization_msgs::MarkerArray>("self_v_marker",1);
  pub_detection = nh.advertise<visualization_msgs::MarkerArray>("detection_marker", 1);
  
  pub_jsk_bbox = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("jsk_output_global",1);

  // tf::TransformListener listener;
  tf_listener = new tf::TransformListener();


  ros::Rate r(10);
  while(ros::ok()){

    ros::spin();
    r.sleep();
  }

  return 0;
}