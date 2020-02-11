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
#include "kf_tracker/featureDetection.h"
#include "kf_tracker/CKalmanFilter.h"
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Int32MultiArray.h>

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

using namespace std;
using namespace cv;
#define iteration 30 //plan segmentation #


//clusterCentroids -> cens
//prevClusterCenters -> pre_cens

// typedef pcl::PointXYZRGBNormal PointT;
typedef pcl::PointXYZI PointT;
ros::Subscriber sub;
ros::Publisher pub,pub_or;
ros::Publisher pub_voxel,pub_marker;

ros::Publisher plane_filtered_pub;
ros::Publisher cluster_pub;

ros::Publisher mea_now,predict_pub,correct_pub;

//get.pub image
image_transport::Subscriber img_sub;
image_transport::Publisher img_pub;


pcl::PointCloud<PointT>::Ptr cloud_pcl=boost::make_shared <pcl::PointCloud<PointT>> ();
pcl::PointCloud<PointT>::Ptr cloud_pcl_whole=boost::make_shared <pcl::PointCloud<PointT>> ();
sensor_msgs::PointCloud2  output;
pcl::PointCloud<PointT>::Ptr cloud_f=boost::make_shared <pcl::PointCloud<PointT>> ();
//pcl::PointCloud<PointT>::Ptr cloud_filtered;
pcl::PointCloud<PointT>::Ptr cloud_filtered=boost::make_shared <pcl::PointCloud<PointT>> ();
vector<pcl::PointXYZI> cens;
vector<geometry_msgs::Point> pre_cens;
visualization_msgs::MarkerArray m_s,l_s;
int max_size = 0;

////////////////////////kalman/////////////////////////
#define cluster_num 40 
ros::Publisher objID_pub;
// KF init
int stateDim=4;// [x,y,v_x,v_y]//,w,h]
int measDim=2;// [z_x,z_y//,z_w,z_h]
int ctrlDim=0;// control input 0(acceleration=0,constant v model)
std::vector<pcl::PointCloud<PointT>::Ptr > cluster_vec;
ros::Publisher pub_cluster[cluster_num];

cv::KalmanFilter KF[cluster_num];//(stateDim,measDim,ctrlDim,CV_32F);

//std::vector<geometry_msgs::Point> prevClusterCenters;
cv::Mat state(stateDim,1,CV_32F);
cv::Mat_<float> measurement(2,1);
std::vector<int> objID; // Output of the data association using KF
                        // measurement.setTo(Scalar(0));
bool firstFrame=true;



// calculate euclidean distance of two points
double euclidean_distance(geometry_msgs::Point& p1, geometry_msgs::Point& p2)
{
  return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}


std::pair<int,int> findIndexOfMin(std::vector<std::vector<float> > distMat)
{
    cout<<"findIndexOfMin CALLED\n";
    std::pair<int,int>minIndex;
    float minEl=std::numeric_limits<float>::max();
    cout<<"minEl="<<minEl<<"\n";

    //第i row為KF[i]與第j群的距離 挑選所有距離中最近的kf跟cluster
    for (int i=0; i<distMat.size();i++)
        for(int j=0;j<distMat.at(0).size();j++)
        {
            if( distMat[i][j]<minEl)
            {
                minEl=distMat[i][j];
                minIndex=std::make_pair(i,j);

            }

        }
    cout<<"minIndex="<<minIndex.first<<","<<minIndex.second<<"\n";
    return minIndex;
}

/* 
void publish_cloud(ros::Publisher& pub, pcl::PointCloud<pcl::PointXYZ>::Ptr cluster){
    sensor_msgs::PointCloud2::Ptr clustermsg (new sensor_msgs::PointCloud2);
    pcl::toROSMsg (*cluster , *clustermsg);
    clustermsg->header.frame_id = "/map";
    clustermsg->header.stamp = ros::Time::now();
    pub.publish (*clustermsg);

}*/


void KFT(const std_msgs::Float32MultiArray ccs)
{
    //std::vector<cv::Mat> pred{KF0.predict(),KF1.predict(),KF2.predict(),KF3.predict(),KF4.predict(),KF5.predict()};
    std::vector<cv::Mat> pred;
    for(int i=0 ;i<cluster_num ;i++){
      pred.push_back(KF[i].predict());
    }
    // Get measurements
    // Extract the position of the clusters forom the multiArray. To check if the data
    // coming in, check the .z (every third) coordinate and that will be 0.0
    std::vector<geometry_msgs::Point> clusterCenters;//clusterCenters
   
    int i=0;
    //convert cens from PointXYZI to geometry_msgs
    cout << "Now cen is:"<<endl;
    /* for (std::vector<float>::const_iterator it=ccs.data.begin();it!=ccs.data.end();it+=3)
    {
        geometry_msgs::Point pt;
        pt.x=*it;
        pt.y=*(it+1);
        pt.z=*(it+2);
        cout <<"("<<pt.x<<","<<pt.y<<","<<pt.z<<")"<<endl;
        clusterCenters.push_back(pt);
    }*/

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
    for (auto it=pred.begin();it!=pred.end();it++)
    {
        geometry_msgs::Point pt;
        pt.x=(*it).at<float>(0);
        pt.y=(*it).at<float>(1);
        pt.z=(*it).at<float>(2);
        cout <<"("<<pt.x<<","<<pt.y<<","<<pt.z<<")"<<endl;
        KFpredictions.push_back(pt);
        predict_pub.publish(pt);
    }

    // Find the cluster that is more probable to be belonging to a given KF.
    objID.clear();//Clear the objID vector
    objID.resize(cluster_num);//Allocate default elements so that [i] doesnt segfault. Should be done better
    // Copy clusterCentres for modifying it and preventing multiple assignments of the same ID
    std::vector<geometry_msgs::Point> copyOfClusterCenters(clusterCenters);
    std::vector<std::vector<float> > distMat;

    for(int filterN=0;filterN<cluster_num;filterN++)
    {
        std::vector<float> distVec;
        for(int n=0;n<cluster_num;n++)
        {
            distVec.push_back(euclidean_distance(KFpredictions[filterN],copyOfClusterCenters[n]));
        }

        distMat.push_back(distVec);
     
     cout<<"filterN="<<filterN<<"\n";

    }

    //cout<<"distMat.size()"<<distMat.size()<<"\n";
    //cout<<"distMat[0].size()"<<distMat.at(0).size()<<"\n";
    // DEBUG: print the distMat
    for ( const auto &row : distMat )
    {
        for ( const auto &s : row ) std::cout << s << ' ';
        std::cout << std::endl;
    }


  //establish link
    for(int clusterCount=0;clusterCount<cluster_num;clusterCount++)
    {
        // 1. Find min(distMax)==> (i,j);
        std::pair<int,int> minIndex(findIndexOfMin(distMat));
         cout<<"Received minIndex="<<minIndex.first<<","<<minIndex.second<<"\n";
        // 2. objID[i]=clusterCenters[j]; counter++
        objID[minIndex.first]=minIndex.second;
    
        // 3. distMat[i,:]=10000; distMat[:,j]=10000
        distMat[minIndex.first]=std::vector<float>(cluster_num,10000.0);// Set the row to a high number.
        for(int row=0;row<distMat.size();row++)//set the column to a high number
        {
            distMat[row][minIndex.second]=10000.0;
        }
        // 4. if(counter<6) got to 1.
        cout<<"clusterCount="<<clusterCount<<"\n";

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
    cout << m_s.markers.size() <<endl;
    m_s.markers.resize(cluster_num);
    m_s.markers.clear();
    cout<< m_s.markers.size()<< endl;

    int k;
    visualization_msgs::Marker marker;
    for(k=0; k<cens.size(); k++){
        marker.header.frame_id="/map";
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
        //stringstream ss;
        //ss << k;
        for(int a=0; a<cluster_num; a++){
          if( objID[a] == k ){
            stringstream ss;
            ss << a;
            // marker.color.b = 0.0f;
            // marker.color.g = 0.0f;
            // marker.color.r = 0.5f + a*0.01f;
            marker.text = ss.str();
            marker.color.a = 1;
            break;
          }
          else
          {
              marker.color.a = 0;
          }
        }
        marker.pose = pose;
        m_s.markers.push_back(marker);
    }

    //if (m_s.markers.size() > max_size)
    //    max_size = m_s.markers.size();

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
///////////////////////////////////////////////////
    /* 
    visualization_msgs::MarkerArray clusterMarkers;

    for (int i=0;i<cluster_num;i++)
    {
      visualization_msgs::Marker m;

      m.id=i;
      m.type=visualization_msgs::Marker::CUBE;
      m.header.frame_id="/map";
      m.scale.x=0.3;         m.scale.y=0.3;         m.scale.z=0.3;
      m.action=visualization_msgs::Marker::ADD;
      m.color.a=1.0;
      m.color.r=i%2?1:0;
      m.color.g=i%3?1:0;
      m.color.b=i%4?1:0;

      //geometry_msgs::Point clusterC(clusterCenters.at(objID[i]));
      geometry_msgs::Point clusterC(KFpredictions[i]);
      m.pose.position.x=clusterC.x;
      m.pose.position.y=clusterC.y;
      m.pose.position.z=clusterC.z;

      clusterMarkers.markers.push_back(m);
    }
    */
    pre_cens.clear();
    pre_cens=clusterCenters;
    //cout << "\033[1;33mpre_cen is:"<<pre_cens.size()<<"\033[0m"<<endl;

  
    std_msgs::Int32MultiArray obj_id;
    for(auto it=objID.begin();it!=objID.end();it++)
        obj_id.data.push_back(*it);
    // Publish the object IDs
    objID_pub.publish(obj_id);
    // convert clusterCenters from geometry_msgs::Point to floats
    std::vector<std::vector<float> > cc;
    for (int i=0;i<cluster_num;i++)
    {
        vector<float> pt;
        geometry_msgs::Point pt_pub;
        pt.push_back(pt_pub.x = clusterCenters[objID[i]].x);
        pt.push_back(pt_pub.y = clusterCenters[objID[i]].y);
        pt.push_back(pt_pub.z = clusterCenters[objID[i]].z);
        cout <<"\033[1;33m KF["<<i<<"]\033[0m measured at ("<<clusterCenters[objID[i]].x<<","<<clusterCenters[objID[i]].y<<")"<<endl;
        //mea_now.publish(pt_pub);
       
        cc.push_back(pt);
    }
    //ith meas for KF[i] cluster measurement(cens)
    float meas[cluster_num][2];
    for(int i=0;i<cluster_num;i++){
        meas[i][0]=cc[i].at(0);
        meas[i][1]=cc[i].at(1);
    }

    cv::Mat measMat[cluster_num];
    for(int i=0;i<cluster_num;i++){
        measMat[i]=cv::Mat(2,1,CV_32F,meas[i]);
    }

    // The update phase 
    
    // if (!(measMat[0].at<float>(0,0)==0.0f || measMat[0].at<float>(1,0)==0.0f))
    //     Mat estimated0 = KF[0].correct(measMat[0]);
    Mat estimated[cluster_num];
    for(int i=0;i<cluster_num;i++){
        if (!(meas[i][0]==0.0f || meas[i][1]==0.0f)){
            geometry_msgs::Point pt;
            pt.x = meas[i][0];
            pt.y = meas[i][1];
            estimated[i] = KF[i].correct(measMat[i]);
            cout << "The corrected state of "<<i<<"th KF is: "<<estimated[i].at<float>(0)<<","<<estimated[i].at<float>(1)<<endl;
            correct_pub.publish(pt);
        }
        else
        {
            cout <<"NOOOO, "<<i<<"th KF is lost!!!!"<<endl; 
        }
        
    }

  return;
}

///////////////////////////////////////////////////////
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


void initiateKF( void ){
  float dvx=0.01f;
  float dvy=0.01f;
  float dx=1.0f;
  float dy=1.0f;
  float dt=0.1f;//time interval btw state transition(10hz ros spin)
  float sigmaP=0.01;//0.01
  float sigmaQ=0.1;//0.1
  for(int i=0;i<cluster_num;i++){
      // KF[i].transitionMatrix = (Mat_<float>(4, 4) << dx,0,1,0,   0,dy,0,1,  0,0,dvx,0,  0,0,0,dvy);
      //KF[i].transitionMatrix = (Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
      KF[i].transitionMatrix = (Mat_<float>(4, 4) << 1,0,dt,0,   0,1,0,dt,  0,0,1,0,  0,0,0,1);
      cv::setIdentity(KF[i].measurementMatrix);
      setIdentity(KF[i].processNoiseCov, Scalar::all(sigmaP));
      cv::setIdentity(KF[i].measurementNoiseCov, cv::Scalar(sigmaQ));
  }
  return;
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

void callback(const sensor_msgs::PointCloud2 &msg){
    
  if( firstFrame ){
    initiateKF();
  }

  sensor_msgs::PointCloud2 out;
  out = msg;
  out.header.frame_id="/map";
  pub_or.publish(out);
  cout<<"I get the car."<<endl;

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

  // pcl::VoxelGrid<PointT> sor;
  // sor.setInputCloud (cloud_pcl);
  // sor.setLeafSize (0.5f, 0.5f, 0.5f); //0.25 for bag1
  // sor.filter (*cloud_filtered);

  // sensor_msgs::PointCloud2 cloud_filtered_sensor;
  // pcl::toROSMsg(*cloud_filtered,cloud_filtered_sensor);
  // cloud_filtered_sensor.header.frame_id = "/map";
  // pub_voxel.publish(cloud_filtered_sensor);

  //Chose ROI to process
  cloud_pcl = crop(cloud_pcl_whole);

  cout<<"Ready to segmentation."<<endl;
  pcl::ExtractIndices<PointT> extract;
  int i = 0, nr_points = (int) cloud_filtered->points.size ();
  
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
    // extract.setNegative (false);
    // extract.filter (*cloud_p);
    //std::cerr << "PointCloud representing the planar component: " << cloud_p->width * cloud_p->height << " data points." << std::endl;

    //cout<<"I'm extracting."<<endl;
    // Create the filtering object
    extract.setNegative (true);
    extract.filter (*cloud_f);
    cloud_pcl.swap (cloud_f);
    cout<<i<<endl;
  }


  pcl::toROSMsg(*cloud_pcl,output);
  output.header.frame_id="/map";
  pub.publish(output);
////////////////////no downsampling
  pcl::VoxelGrid<PointT> sor;
  sor.setInputCloud (cloud_pcl);
  sor.setLeafSize (0.25f, 0.25f, 0.25f); //0.25 for bag1; 0.1 for 3
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
  
  cens.clear();
  //cens.resize(cluster_num);
  cout << "\033[1;33mCen is:"<<cens.size()<<"\033[0m"<<endl;
  
  
  cluster_vec.clear();
  //cout<< "\033[1;33mcluster_vec is empty(1):\033[0m"<<cluster_vec.empty()<< endl;
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
  
  while (cluster_vec.size() < cluster_num){
    pcl::PointCloud<pcl::PointXYZI>::Ptr empty_cluster (new pcl::PointCloud<pcl::PointXYZI>);
    empty_cluster->points.push_back(pcl::PointXYZI(1.0f));
    cluster_vec.push_back(empty_cluster);
  }

  while (cens.size()<cluster_num)
  {
    pcl::PointXYZI centroid;
    centroid.x=0.0;
    centroid.y=0.0;
    centroid.z=0.0;
    
    cens.push_back(centroid);
  }
  

  if( firstFrame ){
    // Set initial state
    for(int i=0;i<cluster_num;i++){
      KF[i].statePost.at<float>(0)=cens.at(i).x;
      KF[i].statePost.at<float>(1)=cens.at(i).y;
      KF[i].statePost.at<float>(2)=0;// initial v_x
      KF[i].statePost.at<float>(3)=0;//initial v_y
    }

    for (int i=0;i<cluster_num;i++)
    {
      geometry_msgs::Point pt;
      pt.x=cens.at(i).x;
      pt.y=cens.at(i).y;
      pre_cens.push_back(pt);
    }
    //第一次marker,紀錄tag
  //publish cluster markers
  //visualization_msgs::MarkerArray m_s;
    cout << m_s.markers.size() <<endl;
    m_s.markers.clear();
    cout<< "\033[1;33mMarkerarray is empty(1):\033[0m"<<m_s.markers.empty()<< endl;

    int k;
    visualization_msgs::Marker marker;
    for(k=0; k<cens.size(); k++){
      marker.header.frame_id="/map";
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

    //if (m_s.markers.size() > max_size)
    max_size = m_s.markers.size();

//     for (int a = k; a < max_size; a++)
//     {
//         marker.id = a;
//         marker.color.a = 0;
//         marker.pose.position.x = 0;
//         marker.pose.position.y = 0;
//         marker.pose.position.z = 0;
//         marker.scale.z = 0;
//         m_s.markers.push_back(marker);
// //       ++marker_id;
//     }
    pub_marker.publish(m_s);
    
    cloud_clusters = crop(cloud_clusters);
    sensor_msgs::PointCloud2 cluster_cloud;
    pcl::toROSMsg(*cloud_clusters, cluster_cloud);
    cluster_cloud.header.frame_id = "map";
    cluster_pub.publish(cluster_cloud);

          
    firstFrame=false;
    return;//////////////////////////first initialization down 
  }

/////////////第二次scan後才有kft
  std_msgs::Float32MultiArray cc;
  for(int i=0;i<cluster_num;i++)
  {
      cc.data.push_back(cens.at(i).x);
      cc.data.push_back(cens.at(i).y);
      cc.data.push_back(cens.at(i).z);    
  }

  KFT(cc);
                 
  cloud_clusters = crop(cloud_clusters);
  sensor_msgs::PointCloud2 cluster_cloud;
  pcl::toROSMsg(*cloud_clusters, cluster_cloud);
  cluster_cloud.header.frame_id = "map";
  cluster_pub.publish(cluster_cloud);
  
  return;
      /* 
    int i=0;
    bool publishedCluster[cluster_num];
    for(auto it=objID.begin();it!=objID.end();it++)
    { //cout<<"Inside the for loop\n";
        cout<<"Inside the switch case\n";
        if(i<cluster_num){
            publish_cloud(pub_cluster[i],cluster_vec[*it]);
            publishedCluster[i]=true;//Use this flag to publish only once for a given obj ID
            i++;
        }
       
    }*/
}


/* 

void img_callback(const sensor_msgs::ImageConstPtr &msg){//const sensor_msgs::CompressedImageConstPtr &msg
  Mat img;
  cout<<"I'm in img_callback."<<endl;
  img = cv_bridge::toCvShare(msg, "bgr8")->image;
  namedWindow("Raw Img", WINDOW_AUTOSIZE);
  imshow("Raw Img", img);
  //waitKey(10);

  //draw point on image
  int point_size = 1;
  for(int i=0; i<point_2d.size(); i++){
    circle(img, point_2d[i], point_size, Scalar(0,255,0), -1);//BGR
  }
  cout<<"D"<<endl;
  namedWindow("Projected Img", WINDOW_AUTOSIZE);
  imshow("Projected Img", img);
  waitKey(10);
  
  
  return;
}

*/

int main(int argc,char** argv){
    ros::init(argc,argv,"seeya");
    ros::NodeHandle nh;

    sub = nh.subscribe("points_raw",1000,&callback);
    pub = nh.advertise<sensor_msgs::PointCloud2>("/lala",1000);
    pub_or = nh.advertise<sensor_msgs::PointCloud2>("/iu",1000);
    pub_voxel = nh.advertise<sensor_msgs::PointCloud2>("/voxel",1000);

    cluster_pub = nh.advertise<sensor_msgs::PointCloud2 >("cluster_cloud", 1);
    pub_marker = nh.advertise<visualization_msgs::MarkerArray>("marker", 1);
    objID_pub = nh.advertise<std_msgs::Int32MultiArray>("obj_id", 1);

    //mea_now = nh.advertise<geometry_msgs::Point>("now", 1);
    //predict_pub = nh.advertise<geometry_msgs::Point>("pre", 1);
    //correct_pub = nh.advertise<geometry_msgs::Point>("correct", 1);

    for(int i=0 ;i<cluster_num; i++){
      KF[i].init(stateDim,measDim,ctrlDim,CV_32F);
    }

    image_transport::ImageTransport it(nh);
   // img_sub = it.subscribe("zed/left/image_raw_color", 1, img_callback);

    ros::Rate r(10);
    while(ros::ok()){
        ros::spin();
        pub_marker.publish(m_s);
        pub.publish(output);
        r.sleep();
    }

}
