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

using namespace std;
using namespace cv;
#define iteration 20 //plan segmentation #


typedef pcl::PointXYZI PointT;
ros::Subscriber sub,label_sub;
ros::Publisher pub,pub_get;
ros::Publisher pub_voxel,pub_marker;

ros::Publisher plane_filtered_pub;
ros::Publisher cluster_pub;

ros::Publisher mea_now,predict_pub,correct_pub;

//get.pub image
image_transport::Subscriber img_sub;
image_transport::Publisher img_pub;


tf::Transform scan_transform ;
// tf::TransformListener listener;

// pcl::PointCloud<PointT>::Ptr cloud_pcl=boost::make_shared <pcl::PointCloud<PointT>> ();
// pcl::PointCloud<PointT>::Ptr cloud_pcl_whole=boost::make_shared <pcl::PointCloud<PointT>> ();
sensor_msgs::PointCloud2  output, now_scan;
// pcl::PointCloud<PointT>::Ptr cloud_f=boost::make_shared <pcl::PointCloud<PointT>> ();
//pcl::PointCloud<PointT>::Ptr cloud_filtered;
// pcl::PointCloud<PointT>::Ptr cloud_filtered=boost::make_shared <pcl::PointCloud<PointT>> ();

pcl::PointCloud<PointT>::Ptr cloud_pcl(new pcl::PointCloud<PointT>);
pcl::PointCloud<PointT>::Ptr cloud_pcl_whole(new pcl::PointCloud<PointT>);
pcl::PointCloud<PointT>::Ptr cloud_f(new pcl::PointCloud<PointT>);
pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);


vector<PointT> cens, cens_all;
vector<geometry_msgs::Point> pre_cens;
visualization_msgs::MarkerArray m_s,l_s,current_m_a;
int max_size = 0;

float dt = 1.0f;//0.1f
float sigmaP=0.01;//0.01
float sigmaQ=0.1;//0.1
#define bias 3.0 //5
int id_count = 0; //the counter to calculate the current occupied track_id

////////////////////////kalman/////////////////////////
#define cluster_num 40 
#define frame_lost 5
#define detect_thres 2.5 //l=4.5,w=1.8 (4.7,1.86 for nu)

bool get_label = false;

std::vector <string> tracked;
std::vector <int> count,id;
std::vector<geometry_msgs::Point> pred_velocity;
int un_assigned;
std::vector <float> kf_pre_cens;
//std::vector <int> un_assigned_index;

ros::Publisher objID_pub;
// KF init
int stateDim=6;// [x,y,v_x,v_y]//,w,h]
int measDim=3;// [z_x,z_y//,z_w,z_h]
int ctrlDim=0;// control input 0(acceleration=0,constant v model)
std::vector<pcl::PointCloud<PointT>::Ptr> cluster_vec;

std::vector<cv::KalmanFilter> KF;//(stateDim,measDim,ctrlDim,CV_32F);

//std::vector<geometry_msgs::Point> prevClusterCenters;
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
  int match_clus = 1000;
  int cluster_idx;
  int uuid ;
}track;

std::vector<track> filters;

ros::Time lidar_timestamp, label_timestamp;

// calculate euclidean distance of two points
double euclidean_distance(geometry_msgs::Point p1, geometry_msgs::Point p2)
{
  return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}


// first_index -> track_idx, second_index->cluster_idx
// Find the min of distmat, and return indexpair(i,j)
// std::pair<int,int> findIndexOfMin(std::vector<std::vector<float> > distMat)
// {
//     cout<<"findIndexOfMin CALLED\n";
//     std::pair<int,int>minIndex;
//     float minEl=std::numeric_limits<float>::max();
//     cout<<"minEl="<<minEl<<"\n";

//     //第i row為KF[i]與第j群的距離 ,但這樣挑會有順序性：0->1->.....49 KF[0]有選擇優先權,可以設threshold  但要如何設？（若單純用距離又與bag播放速度有關 速度資訊？）
//     for (int i=0; i<distMat.size();i++)
//         for(int j=0;j<distMat.at(0).size();j++)
//         {
//             if( distMat[i][j]<minEl)
//             {
//                 minEl=distMat[i][j];
//                 minIndex=std::make_pair(i,j);

//             }

//         }
//     cout<<"minIndex="<<minIndex.first<<","<<minIndex.second<<"\n";
//     return minIndex;
// }




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
    //c.intensity = j+100;
    cens.push_back(c);
    //cloud_cluster->points.push_back(c);
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
                                            0,0,0,1000.0,0,0,
                                            0,0,0,0,1000.0,0,
                                            0,0,0,0,0,1000.0);

    // cv::setIdentity(ka.errorCovPost);
    // for (int i=3;i<6;i++){
    //   for(int j=3; j<6 ;j++){
    //     if (i==j)
    //      ka.errorCovPost[i][j] = 1000.0; 
    //   }
    // }

    // cout << ka.errorCovPost <<endl;
    
  
    //predict phase to generate statePre( state X(K|K-1) ), so to correct to X(K|K) by measurement 
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

    // uuid_t uu;
    // uuid_generate(uu);
    // for (int k = 0 ; k<16; k++){
    //   uu[i]
    // }
    tk.uuid = ++id_count;
    filters.push_back(tk);
    cout<<"Done init newT at "<<id_count<<" is ("<<tk.pred_pose.x <<"," <<tk.pred_pose.y<<")"<<endl;
    
    
    return tk.uuid;
}



// void initiateKF( void ){
//   float dvx=0.01f;
//   float dvy=0.01f;
//   float dx=1.0f;
//   float dy=1.0f;
//   float dt=0.1f;//time interval btw state transition(10hz ros spin)
//   float sigmaP=0.01;//0.01
//   float sigmaQ=0.1;//0.1
//   for(int i=0;i<cluster_num;i++){
//       // KF[i].transitionMatrix = (Mat_<float>(4, 4) << dx,0,1,0,   0,dy,0,1,  0,0,dvx,0,  0,0,0,dvy);
//       //KF[i].transitionMatrix = (Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
//       KF[i].transitionMatrix = (Mat_<float>(4, 4) << 1,0,dt,0,   0,1,0,dt,  0,0,1,0,  0,0,0,1);
//       cv::setIdentity(KF[i].measurementMatrix);
//       setIdentity(KF[i].processNoiseCov, Scalar::all(sigmaP));
//       cv::setIdentity(KF[i].measurementNoiseCov, cv::Scalar(sigmaQ));
//   }
//   return;
// }


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


//writing template function remove for different types, same implementation function call
// template <class T,U> 
// T remove (T list,U index){
//   T temp;
	
// 	int i=0;
// 	for(int j=0; i<list.size() && j<index.size(); i++){
// 		if(i==index[j]){
// 			j++;
// 		}
// 		else{
// 			temp.push_back(list[i]);
// 		}
// 	}
	
// 	std::copy(std::next(list.begin(), i), list.end(), std::back_inserter(temp));
// 	return temp;
// }



void filter_cluster(){
  cout<<"Filtering the cluster.\n";
  get_label = false;
  for (int i = 0; i<cens_all.size(); i++){
    for (int j = 0; j<current_m_a.markers.size(); j++){
      visualization_msgs::Marker m = current_m_a.markers.at(j);
      float x = m.pose.position.x;
      float y = m.pose.position.y;
      float z = m.pose.position.z;
      geometry_msgs::Point p; //= geometry_msgs::Point(x,y,z);
      p.x = x;
      p.y = y;
      p.z = z;
      geometry_msgs::Point p_cen;//= geometry_msgs::Point(cens_all.at(i).x, cens_all.at(i).y, cens_all.at(i).z);
      p_cen.x = cens_all.at(i).x;
      p_cen.y = cens_all.at(i).y;
      p_cen.z = cens_all.at(i).z;
      double dist = euclidean_distance(p,p_cen);

      if (dist <= detect_thres)
        cens.push_back(cens_all.at(i));

      // double or_x = m.pose.orientation.x;
      // double or_y = m.pose.orientation.y;
      // double or_z = m.pose.orientation.z;
      // double or_w = m.pose.orientation.w;
      // float scale_x = m.scale.x;
      // float scale_y = m.scale.y;
      // float scale_z = m.scale.z;


      // std::vector<PointT> corners;
      // PointT point;
      // float ori_x_max = x + scale_x/2;
      // float ori_x_min = x - scale_x/2;
      // float ori_y_max = y + scale_y/2;
      // float ori_y_min = y - scale_y/2;
      // point.x = ori_x_min;
      // point.y = ori_y_min;
      // point.z = 0;
      // corner.push_back(point);
      // point.x = ori_x_min;
      // point.y = ori_y_max;
      // point.z = 0;
      // corner.push_back(point);
      // point.x = ori_x_max;
      // point.y = ori_y_min;
      // point.z = 0;
      // corner.push_back(point);
      // point.x = ori_x_max;
      // point.y = ori_y_max;
      // point.z = 0;
      // corner.push_back(point);


      // pcl::PointCloud<PointT>::Ptr corner_p, corner_rotated(new pcl::PointCloud<PointT>);
      // for (int k = 0; k<4; k++)
      //   corner_p->points.push_back(corner.at(k));



      // Eigen::Vector3f offset= Eigen::Vector3f(x,y,z);
      // Eigen::Quaternionf orientation = Eigen::Quaternionf(or_x, or_y, or_z, or_w);

      // // pcl::PointCloud<PointT>::Ptr now_scan_pcl(new pcl::PointCloud<PointT>);
      // // pcl::fromROSMsg(now_scan,*now_scan_pcl);
      // pcl::transformPointCloud(*corner_p, *corner_rotated, offset, orientation);

      // float max_x, min_x, max_y, min_y = 0f;
      // for (int l = 0; l<4; l++){
      //   float upper_x = corner_rotated->points[l].x;
      //   if (m_x > max_x)
      //     max_x = m_x;
      //   if ()
      // }

    }
  }
}




void KFT(void)
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
    }

    // Get measurements
    // Extract the position of the clusters forom the multiArray. To check if the data
    // coming in, check the .z (every third) coordinate and that will be 0.0
    std::vector<geometry_msgs::Point> clusterCenters;//clusterCenters
   
    int i=0;
    cout << "Now cen is:"<<endl;
    for(i; i<cens.size(); i++){
      geometry_msgs::Point pt;
      pt.x=cens[i].x;
      pt.y=cens[i].y;
      pt.z=cens[i].z;
      cout <<"("<<pt.x<<","<<pt.y<<","<<pt.z<<")"<<endl;
      clusterCenters.push_back(pt);
    }

    std::vector<geometry_msgs::Point> KFpredictions;
    i=0;
    
    cout<<"--------------------\nThe prediction is:"<<endl;
    for (auto it=filters.begin();it!=filters.end();it++)
    {
        geometry_msgs::Point pt;
        pt.x = (*it).pred_pose.x;
        pt.y = (*it).pred_pose.y;
        pt.z = (*it).pred_pose.z;
        cout << "("<<pt.x<<","<<pt.y<<","<<pt.z<<")"<<endl;
      //   track tk = (*it);
      //   geometry_msgs::Point pt;
      //   pt.x=tk.;
      //   pt.y=(*it).at<float>(1);
      //   pt.z=(*it).at<float>(2);
      //   cout <<"("<<pt.x<<","<<pt.y<<","<<pt.z<<")"<<endl;
      //   KFpredictions.push_back(pt);
      //  // predict_pub.publish(pt);

      //   geometry_msgs::Point pt_v;
      //   pt_v.x=(*it).at<float>(3);
      //   pt_v.y=(*it).at<float>(4);
      //   pt_v.z=(*it).at<float>(5);
      //   pred_velocity.push_back(pt_v);

    }  cout<<"--------------------\nThe pred_v is:"<<endl;
    for (auto it=filters.begin();it!=filters.end();it++)
    {
        geometry_msgs::Point pt;
        pt.x = (*it).pred_v.x;
        pt.y = (*it).pred_v.y;
        pt.z = (*it).pred_v.z;
        cout << "("<<pt.x<<","<<pt.y<<","<<pt.z<<")"<<endl;

    }


    //construct dist matrix (mxn): m tracks, n clusters.
    std::vector<geometry_msgs::Point> copyOfClusterCenters(clusterCenters);
    std::vector<std::vector<double> > distMat; //float

    for(std::vector<track>::const_iterator it = filters.begin (); it != filters.end (); ++it)
    {
        std::vector<double> distVec; //float
        for(int n=0;n<cens.size();n++)
        {
            distVec.push_back(euclidean_distance((*it).pred_pose,copyOfClusterCenters[n]));
        }

        distMat.push_back(distVec);

    }
  
    // DEBUG: print the distMat
    for ( const auto &row : distMat )
    {
        for ( const auto &s : row ) std::cout << s << ' ';
        std::cout << std::endl;
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
      if ( assignment[k] != -1 ){
        if( dist_vec.at(assignment[k]) <=  dist_thres + bias ){//bias as gating function to filter the impossible matched detection 
        
          obj_id[assignment[k]] = filters.at(k).uuid;
          filters.at(k).cluster_idx = assignment[k];
          filters.at(k).state = "tracking";  

          distMat[k]=std::vector<double>(cens.size(),10000.0); //float
          for(int row=0;row<distMat.size();row++)//set the column to a high number
          {
              distMat[row][assignment[k]]=10000.0;
          }  
        }
        else
        {
          filters.at(k).state= "lost";
        } 

      }
      else
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
          cout<<"The state of "<<k<<" filters is \033[1;34m"<<filters.at(k).state<<"\033[0m,to cluster_idx "<< filters.at(k).cluster_idx <<" track: "<<filters.at(k).track_frame<<endl;
      else if (filters.at(k).state== "lost")
          cout<<"The state of "<<k<<" filters is \033[1;34m"<<filters.at(k).state<<"\033[0m,to cluster_idx "<< filters.at(k).cluster_idx <<" lost: "<<filters.at(k).lose_frame<<endl;
      else
          cout<<"\033[1;31mUndefined state for trackd "<<k<<"\033[0m"<<endl;
      
    
    }  

    cout<<"\033[1;33mThe obj_id:<<\033[0m\n";
    for (i=0; i<cens.size(); i++){
      cout<<obj_id.at(i)<<" ";
      }
    cout<<endl;

    //initiate new tracks for not-matched cluster
    for (i=0; i<cens.size(); i++){
      if(obj_id.at(i) == -1){
        int track_uuid = new_track(cens.at(i),i);
        obj_id.at(i) = track_uuid;
      }
    }

    cout<<"\033[1;33mThe obj_id after new track:<<\033[0m\n";
    for (i=0; i<cens.size(); i++){
      cout<<obj_id.at(i)<<" ";
      }
    cout<<endl;

    // for(k; k<cens.size(); k++)
    // {
    //   minIndex = findIndexOfMin(distMat);
    //   cout<<"Received minIndex="<<minIndex.first<<","<<minIndex.second<<"\n";
    //   int track_idx = minIndex.first;
    //   int cluster_idx = minIndex.second;    
    //   geometry_msgs::Point pt = filters.at(track_idx).pred_v;
    //   float pred_dist =  sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z) *dt;
    //   // if( distMat[track_idx][cluster_idx] <= pred_dist + bias ){//但回傳最小 表從某個回傳開始都會loss track 
    //   if( distMat[track_idx][cluster_idx] <=  bias ){//但回傳最小 表從某個回傳開始都會loss track 
    //     obj_id[cluster_idx] = filters.at(track_idx).id;
    //     filters.at(track_idx).cluster_idx = cluster_idx;
    //     filters.at(track_idx).state = "tracking";    
    //   // geometry_msgs::Point pt = (*pit).pred_v;
    //   // float pred_dist =  sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z) *dt;
    //   // if( distMat[minIndex.first][minIndex.second] <= pred_dist + bias ){//但回傳最小 表從某個回傳開始都會loss track 
    //   //   obj_id[minIndex.second] = pit->id;
    //   //   pit->cluster_idx = minIndex.first;
    //   //   pit->state = "tracking";
   
    //     distMat[minIndex.first]=std::vector<float>(cens.size(),10000.0);
    //     for(int row=0;row<distMat.size();row++)//set the column to a high number
    //     {
    //         distMat[row][minIndex.second]=10000.0;
    //     }
    //   }
    //   else{
    //       filters.at(k).state = "lost";
    //   }
                
    // }

    // for(int row=0;row<distMat.size();row++)//set the column to a high number
    //     {
    //         if(distMat[row][minIndex.second] == 10000.0)
    //             continue;
    //         else{
    //             new_track(cens.at(row),row);
    //         }
    //     }

    //deal with lost tracks and let it remain const velocity -> update pred_pose to meas correct
    for(std::vector<track>::iterator pit = filters.begin (); pit != filters.end ();){
        if( !(*pit).state.compare("lost")  )//true for 0
            (*pit).lose_frame += 1;

        if( !(*pit).state.compare("tracking")  ){//true for 0
            (*pit).track_frame += 1;
            (*pit).lose_frame = 0;
        }
        
        if ( (*pit).lose_frame == frame_lost )
            //remove track from filters
            pit = filters.erase(pit);
        else
            pit ++;
            
    }

    cout<<"We now have "<<filters.size()<<" tracks."<<endl;



    //   else //matched
    //   {
    //     un_assigned--;
    //     count[minIndex.first] = 0;//for retracking tracks 
    //     tracked[minIndex.first] = "tracking";
    //     objID[minIndex.first]=minIndex.second;

    //     distMat[minIndex.first]=std::vector<float>(current_id,10000.0);// Set the row to a high number.
    //     for(int row=0;row<distMat.size();row++)//set the column to a high number
    //     {
    //         distMat[row][minIndex.second]=10000.0;
    //     }
    //   }

    //call remove to delete lost track in every vector to maintain index consistency
    // KF = remove(KF,lost_index);
    // count = remove(count,lost_index);
    // KFpredictions = remove(KFpredictions,lost_index);
    // objID = remove(objID,lost_index);



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
    // cout<< m_s.markers.size()<< endl;

    k=0;
    visualization_msgs::Marker marker;
    for(k; k<cens.size(); k++){
        marker.header.frame_id="/nuscenes_lidar";
        marker.header.stamp = ros::Time();//to show every tag  (ros::Time::now()for latest tag)
        //marker.ns = "basic_shapes";
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.id = k;
        marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;

        marker.scale.z = 2.0f;
        marker.color.b = 1.0f;
        marker.color.g = 1.0f;
        marker.color.r = 1.0f;
        marker.color.a = 1;

        geometry_msgs::Pose pose;
        pose.position.x = cens[k].x;
        pose.position.y = cens[k].y;
        pose.position.z = cens[k].z+2.0f;
        
        //-----------first frame要先發佈tag 為initial 
        stringstream ss;
        ss << obj_id.at(k);
        marker.text = ss.str();
        
        // for(std::vector<track>::const_iterator it = filters.begin (); it != filters.end (); ++it){
        //     if( (*it).cluster_idx == k ){
        //       stringstream ss;
        //       ss << (*it).id;
        //         // string id = (string)(*it).id;
        //         // stringstream ss;
        //         // ss << ;
        //         // marker.color.b = 0.0f;
        //         // marker.color.g = 0.0f;
        //         // marker.color.r = 0.5f + a*0.01f;
             
        //       marker.color.a = 1;
        //       break;
        //     }
        // }
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
        cout << "The corrected state of "<<i<<"th KF, id = "<<(*it).uuid<<" is "<<estimated[i].at<float>(0)<<","<<estimated[i].at<float>(1)<<endl;
        i++;
    }

  return;
}





void callback(const sensor_msgs::PointCloud2 &msg){
  lidar_timestamp = msg.header.stamp;
  now_scan = msg;  
  sensor_msgs::PointCloud2 out;
  out = msg;
  out.header.frame_id="/map";
  pub_get.publish(out);
  cout<<"I get the car. at "<<lidar_timestamp<<endl;

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
  output.header.frame_id="/map";
  pub.publish(output);
////////////////////no downsampling
  pcl::VoxelGrid<PointT> sor;
  sor.setInputCloud (cloud_pcl);
  sor.setLeafSize (0.35f, 0.35f, 0.35f); //0.25 for bag1; 0.1 for 3
  sor.filter (*cloud_filtered);

  sensor_msgs::PointCloud2 cloud_filtered_sensor;
  pcl::toROSMsg(*cloud_filtered,cloud_filtered_sensor);
  cloud_filtered_sensor.header.frame_id = "/map";
  pub_voxel.publish(cloud_filtered_sensor);



  //cluster
  pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI>);
  tree->setInputCloud (cloud_filtered);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;

  ec.setClusterTolerance (0.45); //0.5->cars merged to one cluster
  ec.setMinClusterSize (25); //30 for bag1
  ec.setMaxClusterSize (400); //300 for bag1
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
  
  cout<< "\033[1;33mcluster_vec size is:\033[0m"<<cluster_vec.size()<< endl;
  cout<< "cens_all ="<<cens_all.size()<<endl;
  cout<< "cens ="<<cens.size()<<endl;
  cout << get_label << endl;
  /////////////////////////////done cluster
  
  // if ( get_label ){
  //   float time_dif = lidar_timestamp.toSec() - label_timestamp.toSec();
  //   if (fabs(time_dif) < 0.2){
  //     cout <<" Differ time"<<fabs(time_dif) << endl;
  //     filter_cluster();
  //   }
  //   else
  //     return;
  // }  
  // else
  //   return;

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
        cout<<"( "<<cens.at(i).x<<","<<cens.at(i).y<<","<<cens.at(i).z<<")\n";
        ka.statePost.at<float>(0)=cens.at(i).x;
        ka.statePost.at<float>(1)=cens.at(i).y;
        ka.statePost.at<float>(2)=cens.at(i).z;
        ka.statePost.at<float>(3)=0;// initial v_x
        ka.statePost.at<float>(4)=0;// initial v_y
        ka.statePost.at<float>(5)=0;// initial v_z
          // ka.statePost.at<float>(5)=0;// initial v_z

        ka.errorCovPost = (Mat_<float>(6, 6) << 1,0,0,0,0,0,
                                          0,1,0,0,0,0,
                                          0,0,1,0,0,0,
                                          0,0,0,1000.0,0,0,
                                          0,0,0,0,1000.0,0,
                                          0,0,0,0,0,1000.0);

        tk.kf = ka;
        tk.state = "tracking";
        tk.lose_frame = 0;
        tk.track_frame = 1;

        // uuid_t uu;
        // uuid_generate(uu);
        tk.uuid = i;
        id_count = i;
        filters.push_back(tk);
    }

    cout<<"Initiate "<<filters.size()<<" tracks."<<endl;

    cout << m_s.markers.size() <<endl;
    m_s.markers.clear();
    cout<< "\033[1;33mMarkerarray is empty(1):\033[0m"<<m_s.markers.empty()<< endl;

    int k;
    visualization_msgs::Marker marker;
    for(k=0; k<cens.size(); k++){
      marker.header.frame_id="/nuscenes_lidar"; //child of pointcloud of tf map
      marker.header.stamp = ros::Time();//to show every tag  (ros::Time::now()for latest tag)
      //marker.ns = "basic_shapes";
      marker.action = visualization_msgs::Marker::ADD;
      marker.pose.orientation.w = 1.0;
      marker.id = k;
      marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;

      marker.scale.z = 2.0f;
      marker.color.b = 1.0f;
      marker.color.g = 1.0f;
      marker.color.r = 1.0f;
      marker.color.a = 1;

      geometry_msgs::Pose pose;
      pose.position.x = cens[k].x;
      pose.position.y = cens[k].y;
      pose.position.z = cens[k].z+2.0f;
      stringstream ss;
      ss << k;
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
//       ++marker_id;
    }
    pub_marker.publish(m_s);
    
    //cloud_clusters = crop(cloud_clusters);
    sensor_msgs::PointCloud2 cluster_cloud;
    pcl::toROSMsg(*cloud_clusters, cluster_cloud);
    cluster_cloud.header.frame_id = "/nuscenes_lidar";
    cluster_pub.publish(cluster_cloud);

          
    firstFrame=false;
    return;//////////////////////////first initialization down 
  }

  KFT();
                 
  cloud_clusters = crop(cloud_clusters);
  sensor_msgs::PointCloud2 cluster_cloud;
  pcl::toROSMsg(*cloud_clusters, cluster_cloud);
  cluster_cloud.header.frame_id = "/nuscenes_lidar";
  cluster_pub.publish(cluster_cloud);
  
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
  ros::NodeHandle nh;

  // sub = nh.subscribe("points_raw",1000,&callback);
  sub = nh.subscribe("nuscenes_lidar",1,&callback);
  label_sub = nh.subscribe("lidar_label",1,&callback_label);
  
  pub_get = nh.advertise<sensor_msgs::PointCloud2>("original_pc", 1000);
  pub = nh.advertise<sensor_msgs::PointCloud2>("raw_pc", 1000);
  pub_voxel = nh.advertise<sensor_msgs::PointCloud2>("voxel_pc", 1000);

  cluster_pub = nh.advertise<sensor_msgs::PointCloud2>("cluster_pc", 1000);
  pub_marker = nh.advertise<visualization_msgs::MarkerArray>("marker", 1);



  ros::Rate r(10);
  while(ros::ok()){
    ros::spinOnce();
    pub_marker.publish(m_s);
    pub.publish(output);
    r.sleep();
  }

  return 0;
}