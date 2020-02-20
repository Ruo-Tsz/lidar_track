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

ofstream outputFile;


using namespace std;
using namespace cv;
#define iteration 40 //plan segmentation #
#define sync_time 0.09 //10hz => 0.1sec, original 0.2s => one frame shift
string FRAME="/scan";

typedef pcl::PointXYZI PointT;
ros::Subscriber sub,label_sub;
ros::Publisher pub,pub_get,pub_colored;
ros::Publisher pub_voxel,pub_marker,pub_tra,pub_pt,pub_v;

ros::Publisher plane_filtered_pub;
ros::Publisher cluster_pub;

ros::Publisher mea_now,predict_pub,correct_pub;

//get.pub image
image_transport::Subscriber img_sub;
image_transport::Publisher img_pub;


tf::StampedTransform scan_transform;
sensor_msgs::PointCloud2  output, now_scan;
// tf::TransformListener listener;//declare in global would act like initialing a handler, it then before ros::init() and cause error 
tf::TransformListener *tf_listener; //Use a pointer to initialize in main instead

pcl::PointCloud<PointT>::Ptr cloud_pcl(new pcl::PointCloud<PointT>);
pcl::PointCloud<PointT>::Ptr cloud_pcl_whole(new pcl::PointCloud<PointT>);
pcl::PointCloud<PointT>::Ptr cloud_f(new pcl::PointCloud<PointT>);
pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);


vector<PointT> cens, cens_all;
vector<geometry_msgs::Point> pre_cens;
visualization_msgs::MarkerArray m_s,l_s,current_m_a, tra_array, point_array, v_array;
int max_size = 0;
int t_max_size = 0;

float dt = 0.1f;//0.1f
float sigmaP=0.01;//0.01
float sigmaQ=0.1;//0.1
int id_count = 0; //the counter to calculate the current occupied track_id

////////////////////////kalman/////////////////////////
#define frame_lost 5
#define detect_thres 2.5 //l=4.5,w=1.8 (4.7,1.86 for nu)
#define bias 3.0 //5
#define moving 1 //(10hz(0.1s) v>1m/s=60m/min=3.6km/hr = pedestrian)
#define invalid_v 30 // 108km/hr
#define show_tra 6

bool get_label = false;

std::vector <string> tracked;
std::vector <int> count,id;
std::vector<geometry_msgs::Point> pred_velocity;
int un_assigned;
std::vector <float> kf_pre_cens;
//std::vector <int> un_assigned_index;

ros::Publisher objID_pub;
// KF init
int stateDim=6;// [x,y,z,v_x,v_y,v_z]  + w.h.l.theta
int measDim=3;// [x,y,z]
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
}track;


std::vector<track> filters;

ros::Time label_timestamp;


vector<sensor_msgs::PointCloud2> lidar_msgs;

string param,global;




// calculate euclidean distance of two points
double euclidean_distance(geometry_msgs::Point p1, geometry_msgs::Point p2)
{
  return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}



int find_matching(std::vector<float> dist_vec){
  float now_min = std::numeric_limits<float>::max();
  int cluster_idx;
  for (int i=0; i<dist_vec.size(); i++){
    if(dist_vec.at(i)<now_min){
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


int new_track(PointT cen, int idx){
    track tk;
    cv::KalmanFilter ka;
    ka.init(stateDim,measDim,ctrlDim,CV_32F);
    ka.transitionMatrix = (Mat_<float>(6, 6) << 1,0,0,dt,0,0,
                                                0,1,0,0,dt,0,
                                                0,0,1,0,0,dt,
                                                0,0,0,1,0,0,
                                                0,0,0,0,1,0,
                                                0,0,0,0,0,1);
    cv::setIdentity(ka.measurementMatrix);
    cv::setIdentity(ka.processNoiseCov, Scalar::all(sigmaP));
    cv::setIdentity(ka.measurementNoiseCov, cv::Scalar(sigmaQ));
    ka.statePost.at<float>(0)=cen.x;
    ka.statePost.at<float>(1)=cen.y;
    ka.statePost.at<float>(2)=cen.z;
    ka.statePost.at<float>(3)=0;// initial v_x
    ka.statePost.at<float>(4)=0;// initial v_y
    ka.statePost.at<float>(5)=0;// initial v_z


    ka.errorCovPost = (Mat_<float>(6, 6) << 1,0,0,0,0,0,
                                            0,1,0,0,0,0,
                                            0,0,1,0,0,0,
                                            0,0,0,10.0,0,0,
                                            0,0,0,0,10.0,0,
                                            0,0,0,0,0,10.0);


    
    
    //predict phase to generate statePre( state X(K|K-1) ), to correct to X(K|K) 
    geometry_msgs::Point pt, pt_v;
    cv::Mat pred;
    pred = ka.predict();
    pt.x = pred.at<float>(0);
    pt.y = pred.at<float>(1);
    pt.z = pred.at<float>(2);
    pt_v.x = pred.at<float>(3);
    pt_v.y = pred.at<float>(4);
    pt_v.z = pred.at<float>(5);
    tk.pred_pose = pt;
    tk.pred_v = pt_v;
    

    tk.kf = ka;
    tk.state = "tracking";
    tk.lose_frame = 0;
    tk.track_frame = 0;
    tk.cluster_idx = idx;

    tk.uuid = ++id_count;
    filters.push_back(tk);
    // double velocity;
    // velocity = sqrt(pt_v.x*pt_v.x + pt_v.y*pt_v.y + pt_v.z*pt_v.z);
    cout<<"Done init newT at "<<id_count<<endl;
    
    
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
    for(k=0; k<cens.size(); k++){
    //   marker.header.frame_id="/nuscenes_lidar"; //child of pointcloud of tf map
      // marker.header.frame_id="/scan";
      // if ( !(global.compare("global")) )
      //   marker.header.frame_id="/map"; // coordinate transformed by tf to global map frame   //////////back TO MAP to get transformed global view in map!!
      // else
      //   marker.header.frame_id="/scan";
      marker.header.frame_id = FRAME;  
      marker.header.stamp = ros::Time();//to show every tag  (ros::Time::now()for latest tag)
      marker.action = visualization_msgs::Marker::ADD;
      marker.pose.orientation.w = 1.0;
      marker.id = k;
      marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;

      marker.scale.z = 1.2f;
      // marker.color.b = 1.0f;
      // marker.color.g = 1.0f;
      // marker.color.r = 0;
      marker.color.b = 0.0f;//yellow
      marker.color.g = 0.9f;
      marker.color.r = 0.9f;
      marker.color.a = 1;

      geometry_msgs::Pose pose;
      pose.position.x = cens[k].x;
      pose.position.y = cens[k].y;
      pose.position.z = cens[k].z+1.0f;
     
      stringstream ss;
      ss << obj_id.at(k);
      
      marker.text = ss.str();
      marker.pose = pose;
      m_s.markers.push_back(marker);
      }

    if (m_s.markers.size() > max_size)
      max_size = m_s.markers.size();

    for (int a = k; a < max_size; a++)
    {
        marker.id = a;
        marker.color.a = 0;
        marker.pose.position.x = 0;
        marker.pose.position.y = 0;
        marker.pose.position.z = 0;
        marker.scale.z = 0;
        m_s.markers.push_back(marker);
    }
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

    // if (velocity >= moving && !(filters.at(k).state.compare("tracking"))){
    // if (velocity >= moving && velocity <= 5 && !(filters.at(k).state.compare("tracking"))){  
    if (velocity >= moving && velocity < invalid_v && !(filters.at(k).state.compare("tracking"))){  
      visualization_msgs::Marker marker, P;
      marker.header.frame_id = FRAME;  
      marker.header.stamp = ros::Time();//to show every tag  (ros::Time::now()for latest tag)
      marker.action = visualization_msgs::Marker::ADD;
      marker.lifetime = ros::Duration();
      marker.pose.orientation.w = 1.0;
      marker.id = k;
      marker.type = visualization_msgs::Marker::LINE_STRIP;

      marker.scale.x = 0.1f;
      // marker.color.b = 1.0f;
      marker.color.g = 0.9f;
      // marker.color.r = 1.0f;
      marker.color.a = 1;


      P.header.frame_id = FRAME;
      P.header.stamp = ros::Time();
      P.action = visualization_msgs::Marker::ADD;
      P.lifetime = ros::Duration();
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
        for (vector<geometry_msgs::Point>::const_reverse_iterator r_iter = filters.at(k).history.rbegin(); r_iter != filters.at(k).history.rbegin()+ show_tra; ++r_iter){
            geometry_msgs::Point pt = *r_iter;
            marker.points.push_back(pt);
            P.points.push_back(pt);
        }
      }
    
      tra_array.markers.push_back(marker);
      point_array.markers.push_back(P);
    }
  }

  cout<<"We have "<<tra_array.markers.size()<< " moving objects."<<endl;

  pub_tra.publish(tra_array);
  pub_pt.publish(point_array);
}



void show_velocity(){
  cout << "drawing v" << endl;
  for (int k=0; k<filters.size(); k++){

    geometry_msgs::Point pred;
    pred.x = filters.at(k).pred_v.x;
    pred.y = filters.at(k).pred_v.y;
    pred.z = filters.at(k).pred_v.z;
    float velocity = sqrt(pred.x*pred.x + pred.y*pred.y + pred.z*pred.z);

    if ( velocity >= moving && velocity < invalid_v && !(filters.at(k).state.compare("tracking")) ){
      visualization_msgs::Marker arrow;
      arrow.header.frame_id = FRAME;
      arrow.header.stamp = ros::Time();
      arrow.lifetime = ros::Duration(3);
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
      arrow.scale.x = 0.2f;
      arrow.scale.y = 0.4f;

      v_array.markers.push_back(arrow);
    }
  }

  pub_v.publish(v_array);
}


void KFT(ros::Time lidar_timestamp)
{
    // std::vector<cv::Mat> pred;
    for(int i=0 ;i<filters.size() ;i++){
      geometry_msgs::Point pt, pt_v;
      cv::Mat pred;
      cv::KalmanFilter k = filters.at(i).kf;  
      pred = k.predict();
      pt.x = pred.at<float>(0);
      pt.y = pred.at<float>(1);
      pt.z = pred.at<float>(2);
      pt_v.x = pred.at<float>(3);
      pt_v.y = pred.at<float>(4);
      pt_v.z = pred.at<float>(5);
      filters.at(i).pred_pose = pt;
      filters.at(i).pred_v = pt_v;

      float velocity;
      velocity = sqrt(pt_v.x*pt_v.x + pt_v.y*pt_v.y + pt_v.z*pt_v.z);
      
      cout<<"The v for "<<filters.at(i).uuid<<" is " << velocity << endl;
    }

    // Get measurements
    // Extract the position of the clusters forom the multiArray. To check if the data
    // coming in, check the .z (every third) coordinate and that will be 0.0
    std::vector<geometry_msgs::Point> clusterCenters;//clusterCenters
   
    int i=0;
    // cout << "Now cen is:"<<endl;
    for(i; i<cens.size(); i++){
      geometry_msgs::Point pt;
      pt.x=cens[i].x;
      pt.y=cens[i].y;
      pt.z=cens[i].z;
      // cout <<"("<<pt.x<<","<<pt.y<<","<<pt.z<<")"<<endl;
      clusterCenters.push_back(pt);
    }

    i=0;

    //construct dist matrix (mxn): m tracks, n clusters.
    std::vector<std::vector<double> > distMat; //float

    for(std::vector<track>::const_iterator it = filters.begin (); it != filters.end (); ++it)
    {
        std::vector<double> distVec; //float
        for(int n=0;n<cens.size();n++)
        {
            distVec.push_back(euclidean_distance((*it).pred_pose,clusterCenters[n]));
        }

        distMat.push_back(distVec);

    }



    //hungarian method to optimize(minimize) the dist matrix
    HungarianAlgorithm HungAlgo;
    vector<int> assignment;

    double cost = HungAlgo.Solve(distMat, assignment);

    for (unsigned int x = 0; x < distMat.size(); x++)
		  std::cout << x << "," << assignment[x] << "\t";

	  std::cout << "\ncost: " << cost << std::endl;


    std::vector<int> obj_id(cens.size(),-1); 
    int k=0;
    for(k=0; k<filters.size(); k++){
      std::vector<double> dist_vec = distMat.at(k); //float

      geometry_msgs::Point pred_v = filters.at(k).pred_v;
      float dist_thres = sqrt(pred_v.x * pred_v.x + pred_v.y * pred_v.y); //float
      cout << "The dist_thres for " <<filters.at(k).uuid<<" is "<<dist_thres<<endl;
      
      //-1 for non matched tracks

      // 要更改,紀錄track過去x點, dist(meas - 過去pose) <= dist_thres+bias
      if ( assignment[k] != -1 ){
        if( dist_vec.at(assignment[k]) <=  dist_thres + bias ){//bias as gating function to filter the impossible matched detection 
        
          obj_id[assignment[k]] = filters.at(k).uuid;
          filters.at(k).cluster_idx = assignment[k];
          filters.at(k).state = "tracking";  

          // Hungarian only do once. No need to update dist matrix
          // distMat[k]=std::vector<double>(cens.size(),10000.0); //float
          // for(int row=0;row<distMat.size();row++)//set the column to a high number
          // {
          //     distMat[row][assignment[k]]=10000.0;
          // }  
        }
        // tracks missed and cens candidate is false positive
        else
        {
          filters.at(k).state= "lost";
        } 

      }
      else // tracks missed(not correspondance build)
      {
        filters.at(k).state= "lost";
      }
      

    // std::vector<int> obj_id(cens.size(),-1); //record track_uuid for every cluster, -1 for not matched
  
    // //
    // int k=0;
    // int cluster_idx = -1;
    // std::pair<int,int> minIndex;
    // //every track to find the min value

    // for(k=0; k<filters.size(); k++){
    //   std::vector<float> dist_vec = distMat.at(k);

    //   cluster_idx = find_matching(dist_vec);

    //   //////original methon to match nearest neighbor(order-dependent)
    //   geometry_msgs::Point pred_v = filters.at(k).pred_v;
    //   float dist_thres = sqrt(pred_v.x * pred_v.x + pred_v.y * pred_v.y);
    //   cout << "The dist_thres for " <<filters.at(k).uuid<<" is "<<dist_thres<<endl;
    //   //////


    //   // if( dist_vec[cluster_idx] <=  bias ){//bias as gating function to filter the impossible matched detection 
    //   if( dist_vec[cluster_idx] <=  dist_thres + bias ){//bias as gating function to filter the impossible matched detection 
      
    //     obj_id[cluster_idx] = filters.at(k).uuid;
    //     filters.at(k).cluster_idx = cluster_idx;
    //     filters.at(k).state = "tracking";  

    //     distMat[k]=std::vector<float>(cens.size(),10000.0);
    //     for(int row=0;row<distMat.size();row++)//set the column to a high number
    //     {
    //         distMat[row][cluster_idx]=10000.0;
    //     }  
    //   }
    //   else
    //   {
    //     filters.at(k).state= "lost";
    //   }
      




      //get tracked or lost
      if (filters.at(k).state== "tracking")
          cout<<"The state of "<<filters.at(k).uuid<<" filters is \033[1;34m"<<filters.at(k).state<<"\033[0m,to cluster_idx "<< filters.at(k).cluster_idx <<" track: "<<filters.at(k).track_frame<<endl;
      else if (filters.at(k).state== "lost")
          cout<<"The state of "<<filters.at(k).uuid<<" filters is \033[1;34m"<<filters.at(k).state<<"\033[0m,to cluster_idx "<< filters.at(k).cluster_idx <<" lost: "<<filters.at(k).lose_frame<<endl;
      else
          cout<<"\033[1;31mUndefined state for tracked "<<k<<"\033[0m"<<endl;
      
    
    }  

    // cout<<"\033[1;33mThe obj_id:<<\033[0m\n";
    // for (i=0; i<cens.size(); i++){
    //   cout<<obj_id.at(i)<<" ";
    //   }
    // cout<<endl;

    //initiate new tracks for unmatched cluster
    for (i=0; i<cens.size(); i++){
      if(obj_id.at(i) == -1){
        int track_uuid = new_track(cens.at(i),i);
        obj_id.at(i) = track_uuid;
      }
    }

    // checking new_track working
    // cout<<"\033[1;33mThe obj_id after new track:<<\033[0m\n";
    // for (i=0; i<cens.size(); i++){
    //   cout<<obj_id.at(i)<<" ";
    //   }
    // cout<<endl;



    // obj_id now is the traking result of all cens, done associate. if remain -1 => false positive(alarm)
    //////output result
    if ( !(param.compare("output")) ){
      for (int i=0; i<obj_id.size(); i++){
        outputFile << lidar_timestamp << "," << obj_id.at(i) << "," << cens.at(i).x << "," << cens.at(i).y << "," << cens.at(i).z << endl; 
        // outputFile << label_timestamp << "," << obj_id.at(i) << "," << cens.at(i).x << "," << cens.at(i).y << "," << cens.at(i).z << endl;//in get_bbox, we don't care lidar point cloud, just get marker and do
      }
    }



    //deal with lost tracks and let it remain const velocity -> update pred_pose to meas correct
    for(std::vector<track>::iterator pit = filters.begin (); pit != filters.end ();){
        if( !(*pit).state.compare("lost")  )//true for 0
            (*pit).lose_frame += 1;

        if( !(*pit).state.compare("tracking")  ){//true for 0
            (*pit).track_frame += 1;
            (*pit).lose_frame = 0;

            //record the tracking trajectory
            geometry_msgs::Point pt_his;
            pt_his.x = cens.at((*pit).cluster_idx).x;
            pt_his.y = cens.at((*pit).cluster_idx).y;
            pt_his.z = cens.at((*pit).cluster_idx).z;
            (*pit).history.push_back(pt_his);

        }
        
        // if we lose track consecutively lost ''frame_lost'' frames, remove track
        if ( (*pit).lose_frame == frame_lost )
            //remove track from filters
            pit = filters.erase(pit);
        else
            pit ++;
            
    }

    cout<<"We now have "<<filters.size()<<" tracks."<<endl;


    // for (vector<track>::const_iterator it = filters.begin(); it != filters.end(); it++){
    //   cout <<"The tra of "<<(*it).uuid<<" is:"<<endl;
    //   for (int i=0; i<(*it).history.size(); i++){
    //     geometry_msgs::Point pt = (*it).history.at(i);
    //     cout<<pt.x<<","<<pt.y<<","<<pt.z<<endl;
    //   }
    //   cout<<"----------------------------------"<<endl;
    // }


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
        cout <<"The tra of "<<filters.at(j).uuid<<" is:"<<endl;
        for (int i=0; i<filters.at(j).history.size(); i++){
          geometry_msgs::Point pt = filters.at(j).history.at(i);
          cout<<pt.x<<","<<pt.y<<","<<pt.z<<endl;
        }
        cout<<"----------------------------------"<<endl;
      }
    }



    //////////////////////////////////link to tag
    // pair <KFpredicion_index,cluster_centers_index（cens)>
    // KFpredtctions[cluster_num(i)]為每個前一群KF[i]預測的下個點的vector
    // predtion index為現在cluster_center_index的預測
    // 又cluster_center_index（i）對應到第cluster_vec[i]群點雲
    // for第i群 KF[i]
    // cluster_vec[i]->points[index]為下個點 （cluster_vec有對應kf?）
    //obg[prediction_index(KF[i]第i個濾波持續追蹤第i群]=現在scan中某群的index

    //begin mark
    // cout << m_s.markers.size() <<endl;
    m_s.markers.resize(cens.size());
    m_s.markers.clear();
    tra_array.markers.clear();
    point_array.markers.clear();
    v_array.markers.clear();
    
    show_id(obj_id);

    show_trajectory();

    show_velocity();

///////////////////////////////////////////////////estimate(update)
 
    int num = filters.size();
    float meas[num][3];
    i = 0;
    for(std::vector<track>::const_iterator it = filters.begin(); it != filters.end(); ++it){
        if ( (*it).state == "tracking" ){
            PointT pt = cens[(*it).cluster_idx];
            meas[i][0] = pt.x;
            meas[i][1] = pt.y;
            meas[i][2] = pt.z;
        }
        else if ( (*it).state == "lost" ){
            meas[i][0] = (*it).pred_pose.x; //+ (*it).pred_v.x;
            meas[i][1] = (*it).pred_pose.y;//+ (*it).pred_v.y;
            meas[i][2] = (*it).pred_pose.z; //+ (*it).pred_v.z;
        }
        else
        {
            std::cout<<"Some tracks state not defined to tracking/lost."<<std::endl;
        }
        
        i++;
    }

    std::cout<<"mesurement record."<<std::endl;
    cv::Mat measMat[num];
    for(int i=0;i<num;i++){
        measMat[i]=cv::Mat(measDim,1,CV_32F,meas[i]);
    }

    // The update phase 
    
    // if (!(measMat[0].at<float>(0,0)==0.0f || measMat[0].at<float>(1,0)==0.0f))
    //     Mat estimated0 = KF[0].correct(measMat[0]);
    cv::Mat estimated[num];
    i = 0;
    for(std::vector<track>::iterator it = filters.begin(); it != filters.end(); ++it){
        estimated[i] = (*it).kf.correct(measMat[i]); 
        // cout << "The corrected state of "<<i<<"th KF, id = "<<(*it).uuid<<" is "<<estimated[i].at<float>(0)<<","<<estimated[i].at<float>(1)<<endl;
        i++;
    }

  return;
}





void callback(const sensor_msgs::PointCloud2 &msg){
  ros::Time lidar_timestamp = msg.header.stamp;
  float time_dif;
  cout<<"\033[1;33mWe now @ "<<lidar_timestamp<<"\033[0m"<<endl;



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
  

  //update not working~~
  // if ( !get_label ){
  //     cout<<"Without label, detect lidar @"<<lidar_timestamp<<", store."<<endl;
  //     lidar_msgs.push_back(msg);
  //     return;
  // }
  // else{
  //     long int label_time = ( label_timestamp.sec%10000 + label_timestamp.nsec/1000000000.0 )*10.0;
  //     long int lidar_time = ( lidar_timestamp.sec%10000 + lidar_timestamp.nsec/1000000000.0 )*10.0;
      
  //     if (label_time != lidar_time){
  //       if(lidar_time > label_time){
  //         for(std::vector<sensor_msgs::PointCloud2>::const_iterator it = lidar_msgs.begin (); it != lidar_msgs.end (); ){
  //           ros::Time temp_m = (*it).header.stamp;
  //           long int temp_time = ( temp_m.sec%10000 + temp_m.nsec/1000000000.0 )*10.0;
  //           if (temp_time != label_time)
  //             it = lidar_msgs.erase(it);
  //           else
  //           {
  //             lidar_timestamp = temp_m;
  //             cout<<"We find previous lidar " << lidar_timestamp <<" ,and now label "<< label_timestamp<<endl;
  //             break;
  //           }
  //         }
  //       }
  //       else
  //       {
  //         cout<<"label is leading over lidar"<<endl;
  //         return;
  //       }
        
  //     }
  //     else
  //     {
  //       cout<<"We have sync time @ lidar "<<lidar_timestamp<<" ,and label "<<label_timestamp<<endl;
  //     }     
  // }
  // cout<<"We now at "<<lidar_timestamp<<" ,and label "<<label_timestamp<<endl;




  
  now_scan = msg;  
  sensor_msgs::PointCloud2 out;
  sensor_msgs::PointCloud out_2,out_transformed_2;
  out = msg;
  // out.header.frame_id="/map";
  // pub_get.publish(out);
  // cout<<"I get the car. at "<<lidar_timestamp<<endl; //(2) on test off

  
  //test transform to global, not succeed yet
  /////////////////////////////////////////////////////////////////////////(1)
  //transform point cloud to map frame by tf
  if ( !(global.compare("global")) ){
    sensor_msgs::convertPointCloud2ToPointCloud(out, out_2);
    try {
        tf_listener->waitForTransform("/map", "/scan", ros::Time(0), ros::Duration(5.0));
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
  // cloud_pcl = crop(cloud_pcl_whole);
  cloud_pcl = cloud_pcl_whole;

  cout<<"Ready to segmentation."<<endl;
  pcl::ExtractIndices<PointT> extract;
  
  int i = 0;
  for(i=0;i<iteration;i++){
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
    //cout<<i<<endl;
  }


  pcl::toROSMsg(*cloud_pcl,output);
  output.header.frame_id = FRAME;
  pub.publish(output);
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
  
  cout<< "cluster_vec size is:"<<cluster_vec.size()<< endl;
  cout<< "cens_all ="<<cens_all.size()<<endl;
  cout<< "cens ="<<cens.size()<<endl;
  cout << get_label << endl;

  sensor_msgs::PointCloud2 colored_pc;
  pcl::toROSMsg(*cloud_clusters, colored_pc);
  colored_pc.header.frame_id=FRAME;
  pub_colored.publish(colored_pc);
  

  /////////////////////////////done cluster
  
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
    int current_id = cens.size();
    // float dt = 0.1f; 
    float sigmaP=0.01;//0.01
    float sigmaQ=0.1;//0.1

    //initialize new tracks(function)
    //state = [x,y,z,vx,vy,vz]
    for(int i=0; i<current_id;i++){
     //try ka.init 
        track tk;
        cv::KalmanFilter ka;
        ka.init(stateDim,measDim,ctrlDim,CV_32F);
        ka.transitionMatrix = (Mat_<float>(6, 6) << 1,0,0,dt,0,0,
                                                    0,1,0,0,dt,0,
                                                    0,0,1,0,0,dt,
                                                    0,0,0,1,0,0,
                                                    0,0,0,0,1,0,
                                                    0,0,0,0,0,1);
        cv::setIdentity(ka.measurementMatrix);
        cv::setIdentity(ka.processNoiseCov, Scalar::all(sigmaP));
        cv::setIdentity(ka.measurementNoiseCov, cv::Scalar(sigmaQ));
        // cout<<"( "<<cens.at(i).x<<","<<cens.at(i).y<<","<<cens.at(i).z<<")\n";
        ka.statePost.at<float>(0)=cens.at(i).x;
        ka.statePost.at<float>(1)=cens.at(i).y;
        ka.statePost.at<float>(2)=cens.at(i).z;
        ka.statePost.at<float>(3)=0;// initial v_x
        ka.statePost.at<float>(4)=0;// initial v_y
        ka.statePost.at<float>(5)=0;// initial v_z

        ka.errorCovPost = (Mat_<float>(6, 6) << 1,0,0,0,0,0,
                                          0,1,0,0,0,0,
                                          0,0,1,0,0,0,
                                          0,0,0,10.0,0,0,
                                          0,0,0,0,10.0,0,
                                          0,0,0,0,0,10.0);

        tk.kf = ka;
        tk.state = "tracking";
        tk.lose_frame = 0;
        tk.track_frame = 1;

        // uuid_t uu;
        // uuid_generate(uu);
        tk.uuid = i;
        id_count = i;

        geometry_msgs::Point pt_his;
        pt_his.x = cens.at(i).x;
        pt_his.y = cens.at(i).y;
        pt_his.z = cens.at(i).z;
        tk.history.push_back(pt_his);

        filters.push_back(tk);
    }

    cout<<"Initiate "<<filters.size()<<" tracks."<<endl;

    m_s.markers.clear();
    // cout<< "\033[1;33mMarkerarray is empty(1):\033[0m"<<m_s.markers.empty()<< endl;
    std::vector<int> obj_id(cens.size(),-1);
    for(int i =0; i<cens.size(); i++){
      obj_id.at(i) = i;
    }
    show_id(obj_id);
    
    //cloud_clusters = crop(cloud_clusters);
    /*1
    sensor_msgs::PointCloud2 cluster_cloud;
    pcl::toROSMsg(*cloud_clusters, cluster_cloud); 
    // cluster_cloud.header.frame_id = "/nuscenes_lidar";
    cluster_cloud.header.frame_id = "/scan"; 
    cluster_pub.publish(cluster_cloud); 
  */
          
    firstFrame=false;
    return;//////////////////////////first initialization down 
  }

  cout << "\033[1;31m"<<cens.size()<<"\033[0m" <<endl;
  KFT(lidar_timestamp);
                 
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




int main(int argc, char** argv){
  ros::init(argc,argv,"tracking");
  ros::NodeHandle nh("~");

  string out_dir = ros::package::getPath("lidar_track");
  // sub = nh.subscribe("points_raw",1000,&callback);
//   sub = nh.subscribe("nuscenes_lidar",1,&callback);
//   label_sub = nh.subscribe("lidar_label",1,&callback_label);
  sub = nh.subscribe("/scan",1,&callback);
  label_sub = nh.subscribe("/anno_marker",1,&callback_label);
  
  
  pub_get = nh.advertise<sensor_msgs::PointCloud2>("original_pc", 1000);
  pub = nh.advertise<sensor_msgs::PointCloud2>("raw_pc", 1000);
  pub_voxel = nh.advertise<sensor_msgs::PointCloud2>("voxel_pc", 1000);
  pub_colored = nh.advertise<sensor_msgs::PointCloud2>("colored_pc",1000);

  cluster_pub = nh.advertise<sensor_msgs::PointCloud2>("cluster_pc", 1);
  pub_marker = nh.advertise<visualization_msgs::MarkerArray>("id_marker", 1);
  pub_tra = nh.advertise<visualization_msgs::MarkerArray>("trajectory_marker",1);
  pub_pt = nh.advertise<visualization_msgs::MarkerArray>("pt_marker", 1);
  pub_v = nh.advertise<visualization_msgs::MarkerArray>("velocity_marker", 1);
  
  

  // tf::TransformListener listener;
  tf_listener = new tf::TransformListener();

  nh.getParam("param", param);
  nh.getParam("global",global);  

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

  ros::Rate r(10);
  while(ros::ok()){

    ros::spinOnce();
    pub_marker.publish(m_s);
    pub.publish(output);
    pub_tra.publish(tra_array);
    pub_pt.publish(point_array);
    pub_v.publish(v_array);
    r.sleep();
  }

  return 0;
}