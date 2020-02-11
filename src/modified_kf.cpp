
using namespace std;
using namespace cv;
#define iteration 30 //plan segmentation #


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
vector<PointT> cens;
vector<geometry_msgs::Point> pre_cens;
visualization_msgs::MarkerArray m_s,l_s;
int max_size = 0;

////////////////////////kalman/////////////////////////
#define cluster_num 40 
#define thres
#define frame_lost
int current_id;
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
cv::Mat_<float> measurement(2,1);
std::vector<int> objID; // Output of the data association using KF
                        // measurement.setTo(Scalar(0));
bool firstFrame=true;

// //////type filter
// typedef struct track{
//   cv::KalmanFilter kf;
//   int loss_count;
//   string state;
//   int match_clus = 1000;
//   int id ;
// }track;

// std::vector<track> filter;


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

    //第i row為KF[i]與第j群的距離 ,但這樣挑會有順序性：0->1->.....49 KF[0]有選擇優先權,可以設threshold  但要如何設？（若單純用距離又與bag播放速度有關 速度資訊？）
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


std::pair<int,int> findIndexOfMin(std::vector<std::vector<float> > distMat, std::vector<geometry_msgs::Point> KFpredictions)
{
    cout<<"findIndexOfMin CALLED\n";
    std::pair<int,int>minIndex;
    float minEl=std::numeric_limits<float>::max();
    cout<<"minEl="<<minEl<<"\n";

    //第i row為KF[i]與第j群的距離 挑選所有距離中最近的kf[i]跟cluster[j]
    for (int i=0; i<distMat.size();i++){
        for(int j=0;j<distMat.at(0).size();j++)
        {
            if( distMat[i][j]<minEl)
            {
                minEl=distMat[i][j];
                minIndex=std::make_pair(i,j);

            }
        }
    }

    cout<<"minIndex="<<minIndex.first<<","<<minIndex.second<<"\n";
    return minIndex;
}

//writing template function remove for different types, same implementation function call
template <class T,U> 
T remove (T list,U index){
  T temp;
	
	int i=0;
	for(int j=0; i<list.size() && j<index.size(); i++){
		if(i==index[j]){
			j++;
		}
		else{
			temp.push_back(list[i]);
		}
	}
	
	std::copy(std::next(list.begin(), i), list.end(), std::back_inserter(temp));
	return temp;
}


//////////////先處理State string 再決定誰要predict
void KFT(void)
{
    std::vector<cv::Mat> pred;
    for(int i=0 ;i<KF.size() ;i++){
      pred.push_back(KF[i].predict());
    }

/*
    ////先把有的cluster配對 剩下new track
    //////////if cens.size() > current_id -> must have new track
    for(int i=curretn_id; i<cens.size(); i++){
        cv::KalmanFilter ka;
        ka.transitionMatrix = (Mat_<float>(4, 4) << 1,0,dt,0,   0,1,0,dt,  0,0,1,0,  0,0,0,1);
        cv::setIdentity(ka.measurementMatrix);
        cv::setIdentity(ka.processNoiseCov, Scalar::all(sigmaP));
        cv::setIdentity(ka.measurementNoiseCov, cv::Scalar(sigmaQ));
        // ka.statePost.at<float>(0)=cens.at(i).x;
        // ka.statePost.at<float>(1)=cens.at(i).y;
        // ka.statePost.at<float>(2)=0;// initial v_x
        // ka.statePost.at<float>(3)=0;//initial v_y
        // KF.push_back(ka);
    }

*/

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
    
    pred_velocity.clear();
    cout<<"--------------------\nThe prediction is:"<<endl;
    for (auto it=pred.begin();it!=pred.end();it++)
    {
        geometry_msgs::Point pt;
        pt.x=(*it).at<float>(0);
        pt.y=(*it).at<float>(1);
        pt.z=(*it).at<float>(2);
        cout <<"("<<pt.x<<","<<pt.y<<","<<pt.z<<")"<<endl;
        KFpredictions.push_back(pt);
       // predict_pub.publish(pt);

        geometry_msgs::Point pt_v;
        pt_v.x=(*it).at<float>(3);
        pt_v.y=(*it).at<float>(4);
        pt_v.z=(*it).at<float>(5);
        pred_velocity.push_back(pt_v);

    }

    // Find the cluster that is more probable to be belonging to a given KF.
    objID.clear();//Clear the objID vector
    objID.resize(KF.size());
    //objID.resize(cluster_num);//Allocate default elements so that [i] doesnt segfault. Should be done better
    // Copy clusterCentres for modifying it and preventing multiple assignments of the same ID
    std::vector<geometry_msgs::Point> copyOfClusterCenters(clusterCenters);
    std::vector<std::vector<float> > distMat;

    for(int filterN=0;filterN<KF.size();filterN++)
    {
        std::vector<float> distVec;
        for(int n=0;n<cens.size();n++)
        {
            distVec.push_back(euclidean_distance(KFpredictions[filterN],copyOfClusterCenters[n]));
        }

        distMat.push_back(distVec);
     
     cout<<"filterN="<<filterN<<"\n";

    }

    // DEBUG: print the distMat
    for ( const auto &row : distMat )
    {
        for ( const auto &s : row ) std::cout << s << ' ';
        std::cout << std::endl;
    }


    //establish link for current existing tracks
    un_assigned = cens.size();
    count.resize(KF.size());
    tracked.resize(KF.size());
    cout<<"\033[1;33m"<<count.sizt()<<"\033[0m"<<endl;

    std::vector<int> lost_index;
    //先考慮 kf<=clu 做完for每個track不是lost就是tracking(會找出每個ka最近的clu配對  判別是否<thres)
    for(int clusterCount=0;clusterCount<KF.size();clusterCount++)
    {
      std::pair<int,int> minIndex(findIndexOfMin(distMat,KFpredictions));
      cout<<"Received minIndex="<<minIndex.first<<","<<minIndex.second<<"\n";
    

      if( distMat[minIndex.first][minIndex.second] > thres ){//但回傳最小 表從某個回傳開始都會loss track 
        count[minIndex.first]++;
        //objID[minIndex.first]=1000.0;//KF not matched
        distMat[minIndex.first]=std::vector<float>(current_id,10000.0);
                
        if( count[minIndex.first] > frame_lost ){
            tracked[minIndex.first] = "lost";
            lost_index.push_back(minIndex.first);//record lost kf index


            // //搬出去for 若動態移 KF.size變化 並非最原始大小 裡面先對lost做記號 外面再刪
            // std::vector<int>::iterator it_c = count.begin()+minIndex.first;
            // count.erase(it_c);
            // std::vector<cv::KalmanFilter>::iterator it_kf = KF.begin()+minIndex.first;
            // KF.erase(it_kf);
            // std::vector<geometry_msgs::Point>::iterator it_p = KFpredictions.begin()+minIndex.first;
            // KFpredictions.erase(it_p);
            // std::vector<int>::iterator it_id = objID.begin()+minIndex.first;
            // objID.erase(it_id);
            // ///////
        }

      }
      else //matched
      {
        un_assigned--;
        count[minIndex.first] = 0;//for retracking tracks 
        tracked[minIndex.first] = "tracking";
        objID[minIndex.first]=minIndex.second;

        distMat[minIndex.first]=std::vector<float>(current_id,10000.0);// Set the row to a high number.
        for(int row=0;row<distMat.size();row++)//set the column to a high number
        {
            distMat[row][minIndex.second]=10000.0;
        }
      }

                 
      cout<<"clusterCount="<<clusterCount<<"\n";

    }

    int pre_track = KF.size();//紀錄上一次track數目
    sort(lost_index.begin(),lost_index.end());//sort in asscending form
    for(int i=0; i<lost_index.size(); i++){
      cout << lost_index.at(i) << " " ;
    }
    cout <<endl;

    //call remove to delete lost track in every vector to maintain index consistency
    KF = remove(KF,lost_index);
    count = remove(count,lost_index);
    KFpredictions = remove(KFpredictions,lost_index);
    objID = remove(objID,lost_index);

    //搬出去for 若動態移 KF.size變化 並非最原始大小 裡面先對lost做記號 外面再刪
    // for(int i=0; i<; i++){
      
    // std::vector<int>::iterator it_c = count.begin()+minIndex.first;
    // count.erase(it_c);
    // std::vector<cv::KalmanFilter>::iterator it_kf = KF.begin()+minIndex.first;
    // KF.erase(it_kf);
    // std::vector<geometry_msgs::Point>::iterator it_p = KFpredictions.begin()+minIndex.first;
    // KFpredictions.erase(it_p);
    // std::vector<int>::iterator it_id = objID.begin()+minIndex.first;
    // objID.erase(it_id);
    // ///////
    // }


  //待處理:若kf <= clu 則最後dis都會是10000(不論kf有沒有track row都設為10000表處理過),另外記錄沒有分到的clu的index
    int *ptr;
    ptr = new int[un_assigned];
    int j=0;
    for(int i=0 ;i<current_id; i++){
      for(int k=0;k<distMat.at(0).size(); k++){
        if (distMat[i][k] != 10000.0){
          for(int row=0;row<distMat.size();row++)//set the column to a high number
          {
            distMat[row][minIndex.second]=10000.0;
          }
          ptr[j++] = k;//k for unmatched cluster
        }
      }
    }

處理disMat中未被配對的群（行），100<x<10000,產生新track
/////////////////////////////////////////////////////////////////////////////////

    for(int i=0; i<un_assigned; i++){
      cv::KalmanFilter ka;
      ka.transitionMatrix = (Mat_<float>(6, 6) << 1,0,0,dt,0,0,
                                                  0,1,0,0,dt,0,
                                                  0,0,1,0,0,dt,
                                                  0,0,0,1,0,0,
                                                  0,0,0,0,1,0,
                                                  0,0,0,0,0,1);
      cv::setIdentity(ka.measurementMatrix);
      cv::setIdentity(ka.processNoiseCov, Scalar::all(sigmaP));
      cv::setIdentity(ka.measurementNoiseCov, cv::Scalar(sigmaQ));
      ka.statePost.at<float>(0)=cens.at(ptr[i]).x;
      ka.statePost.at<float>(1)=cens.at(ptr[i]).y;
      ka.statePost.at<float>(2)=cens.at(ptr[i]).z;
      ka.statePost.at<float>(3)=0;// initial v_x
      ka.statePost.at<float>(4)=0;// initial v_y
      ka.statePost.at<float>(5)=0;// initial v_z
      tracked.push_back("tracking");
      KF.push_back(ka);
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
    float meas[current_id][3];
    for(int i=0;i<current_id;i++){
        meas[i][0]=cc[i].at(0);
        meas[i][1]=cc[i].at(1);
        meas[i][2]=cc[i].at(2);
    }

    cv::Mat measMat[current_id];
    for(int i=0;i<current_id;i++){
        measMat[i]=cv::Mat(3,1,CV_32F,meas[i]);
    }

    // The update phase 
    
    // if (!(measMat[0].at<float>(0,0)==0.0f || measMat[0].at<float>(1,0)==0.0f))
    //     Mat estimated0 = KF[0].correct(measMat[0]);
    Mat estimated[current_id];
    for(int i=0;i<current_id;i++){
        if (tracked[i] == "tracking"){
            // geometry_msgs::Point pt;
            // pt.x = meas[i][0];
            // pt.y = meas[i][1];
            estimated[i] = KF[i].correct(measMat[i]);
            cout << "The corrected state of "<<i<<"th KF is: "<<estimated[i].at<float>(0)<<","<<estimated[i].at<float>(1)<<endl;
            //correct_pub.publish(pt);
        }
        else
        {
            cout <<"NOOOO, "<<i<<"th KF is lost!!!!"<<endl; 
        }
        
    }

  return;
}





void callback(const sensor_msgs::PointCloud2 &msg){
    
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

  //Chose ROI to process
  cloud_pcl = crop(cloud_pcl_whole);

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
  
//   while (cluster_vec.size() < cluster_num){
//     pcl::PointCloud<pcl::PointXYZI>::Ptr empty_cluster (new pcl::PointCloud<pcl::PointXYZI>);
//     empty_cluster->points.push_back(pcl::PointXYZI(1.0f));
//     cluster_vec.push_back(empty_cluster);
//   }

//   while (cens.size()<cluster_num)
//   {
//     pcl::PointXYZI centroid;
//     centroid.x=0.0;
//     centroid.y=0.0;
//     centroid.z=0.0;
    
//     cens.push_back(centroid);
//   }
  

  if( firstFrame ){
    current_id = cluster_vec.size();
    float dt = 0.1f; 
    float sigmaP=0.01;//0.01
    float sigmaQ=0.1;//0.1

    //initialize new tracks(function)
    for(int i=0; i<current_id;i++){
        cv::KalmanFilter ka;
        ka.transitionMatrix = (Mat_<float>(6, 6) << 1,0,0,dt,0,0,
                                                    0,1,0,0,dt,0,
                                                    0,0,1,0,0,dt,
                                                    0,0,0,1,0,0,
                                                    0,0,0,0,1,0,
                                                    0,0,0,0,0,1);
        cv::setIdentity(ka.measurementMatrix);
        cv::setIdentity(ka.processNoiseCov, Scalar::all(sigmaP));
        cv::setIdentity(ka.measurementNoiseCov, cv::Scalar(sigmaQ));
        ka.statePost.at<float>(0)=cens.at(i).x;
        ka.statePost.at<float>(1)=cens.at(i).y;
        ka.statePost.at<float>(2)=cens.at(i).z;
        ka.statePost.at<float>(3)=0;// initial v_x
        ka.statePost.at<float>(4)=0;// initial v_y
        ka.statePost.at<float>(5)=0;// initial v_z
        tracked.push_back("tracking");
        KF.push_back(ka);
        id.push_back(i);
    }
    // Set initial state
    // for(int i=0;i<cluster_num;i++){
    //   KF[i].statePost.at<float>(0)=cens.at(i).x;
    //   KF[i].statePost.at<float>(1)=cens.at(i).y;
    //   KF[i].statePost.at<float>(2)=0;// initial v_x
    //   KF[i].statePost.at<float>(3)=0;//initial v_y
    // }

    for (int i=0;i<current_id;i++)
    {
      geometry_msgs::Point pt;
      pt.x=cens.at(i).x;
      pt.y=cens.at(i).y;
      pt.z=cens.at(i).z;
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
    
    //cloud_clusters = crop(cloud_clusters);
    sensor_msgs::PointCloud2 cluster_cloud;
    pcl::toROSMsg(*cloud_clusters, cluster_cloud);
    cluster_cloud.header.frame_id = "map";
    cluster_pub.publish(cluster_cloud);

          
    firstFrame=false;
    return;//////////////////////////first initialization down 
  }

  KFT();
                 
  cloud_clusters = crop(cloud_clusters);
  sensor_msgs::PointCloud2 cluster_cloud;
  pcl::toROSMsg(*cloud_clusters, cluster_cloud);
  cluster_cloud.header.frame_id = "map";
  cluster_pub.publish(cluster_cloud);
  
  return;

}