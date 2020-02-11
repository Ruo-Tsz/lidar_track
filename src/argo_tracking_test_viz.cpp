#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <json/json.h>
#include <ros/ros.h>
#include <fstream>
#include <vector>
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <algorithm>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <tf/transform_broadcaster.h>

#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h> 
#include <pcl/point_types.h> 
#include <pcl/io/ply_io.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sstream>

ros::Subscriber sub;
ros::Publisher pub_scan,pub_marker, pub_gt;
typedef pcl::PointXYZI PointT; 

using namespace std;
#define rate 5
char dir_lidar[100] = "/home/ee904/ARGO/argoverse-tracking/test/train1_6f153f9c-edc5-389f-ac6f-40705c30d97e/lidar";//1cf38cbe-deec-3675-9313-d736e02d1b58
//train1_6f153f9c-edc5-389f-ac6f-40705c30d97e_lane_1_fixbbox_1_rcnn_1_map_const_v_both_roi
string data_path = "/home/ee904/ARGO/argoverse-tracking/test";
string log_name = "train1_6f153f9c-edc5-389f-ac6f-40705c30d97e";
int num = 0;
int max_size = 0;
vector <string> plys;

vector <unsigned long int> sorted_plys;
visualization_msgs::MarkerArray M_array, M_label;

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


Json::Value arrayObj;
vector <an> annotations;
vector <string> id_record;



void initialize_label(void){
    int i;
    annotations.clear();
    for(i=0; i<arrayObj.size(); i++){
        an temp;
        //geometry_msgs::Point pt;
        temp.center.x = arrayObj[i]["center"]["x"].asDouble();
        temp.center.y = arrayObj[i]["center"]["y"].asDouble();
        temp.center.z = arrayObj[i]["center"]["z"].asDouble();

        //geometry_msgs::Quaternion q;
        temp.rotation.x = arrayObj[i]["rotation"]["x"].asDouble();
        temp.rotation.y = arrayObj[i]["rotation"]["y"].asDouble();
        temp.rotation.z = arrayObj[i]["rotation"]["z"].asDouble();
        temp.rotation.w = arrayObj[i]["rotation"]["w"].asDouble();

        temp.height = arrayObj[i]["height"].asDouble();
        temp.length = arrayObj[i]["length"].asDouble();
        temp.width = arrayObj[i]["width"].asDouble();
        
        //這版本json不支援Long 因此timestamp用double轉換表示
        temp.time_s = arrayObj[i]["timestamp"].asDouble();
        temp.track_id = arrayObj[i]["track_label_uuid"].asString();
        temp.track_class = arrayObj[i]["label_class"].asString();
        temp.occlusion = arrayObj[i]["occlusion"].asInt();
        annotations.push_back(temp);
    }
    cout<<"Done initialize."<<endl;
}

void showAllFiles( const char * dir_name, int shit )
{
	// check the parameter !
	if( NULL == dir_name )
	{
		cout<<" dir_name is null ! "<<endl;
		return;
	}
 
	// check if dir_name is a valid dir
	struct stat s;
	lstat( dir_name , &s );
	if( ! S_ISDIR( s.st_mode ) )
	{
		cout<<"dir_name is not a valid directory !"<<endl;
		return;
	}
	
	struct dirent * filename;    // return value for readdir()
 	DIR * dir;                   // return value for opendir()
	dir = opendir( dir_name );
	if( NULL == dir )
	{
		cout<<"Can not open dir "<<dir_name<<endl;
		return;
	}
	cout<<"Successfully opened the dir !"<<endl;
	
	/* read all the files in the dir ~ */
	while( ( filename = readdir(dir) ) != NULL )
	{
		// get rid of "." and ".."
		if( strcmp( filename->d_name , "." ) == 0 || 
			strcmp( filename->d_name , "..") == 0 ||
            strcmp( filename->d_name , ".ipynb_checkpoints") == 0 )
			continue;
		cout<<filename ->d_name <<endl;
        
        if( shit==0 )
            plys.push_back(filename->d_name);
        
    }   
}


void draw_box(void){
    visualization_msgs::Marker marker, marker_label;

    //checking id
    if (id_record.size() == 0){
        for (int k =0; k<num; k++){
            id_record.push_back(annotations.at(k).track_id);
        }
    }
    else{
        for (int k=0; k<num; k++){
            vector<string>::iterator find_id = find(id_record.begin(), id_record.end(),annotations.at(k).track_id);
            if(find_id == id_record.end())
              id_record.push_back(annotations.at(k).track_id);
        }
    }

    int i;
    for(i=0; i<num; i++){
        an temp = annotations.at(i);
        marker.header.frame_id="/scan";
        //marker.header.stamp = ros::Time();//to show every tag  (ros::Time::now()for latest tag)
        marker.header.stamp = ros::Time();
        marker.ns = temp.track_class;
        
        marker.action = visualization_msgs::Marker::ADD;
        marker.lifetime = ros::Duration(0.2);
        //bbox_marker.frame_locked = true;
        marker.type = visualization_msgs::Marker::CUBE;
        
        marker.id = i;
        
        marker.pose.position.x = temp.center.x;
        marker.pose.position.y = temp.center.y;
        marker.pose.position.z = temp.center.z;

        marker.pose.orientation.x = temp.rotation.x;
        marker.pose.orientation.y = temp.rotation.y;
        marker.pose.orientation.z = temp.rotation.z;
        marker.pose.orientation.w = temp.rotation.w;
        
        marker.scale.x = temp.length;
        marker.scale.y = temp.width;
        marker.scale.z = temp.height;
        
        marker.color.b = 0.0f;
        marker.color.g = 0.0f; //1.0f
        marker.color.r = 1.0f;
	    marker.color.a = 0.7f;

        M_array.markers.push_back(marker);
        //////////////////search for id index

        for(int m=0; m<id_record.size(); m++){
            if ( !(temp.track_id.compare(id_record.at(m))) ) {
                stringstream ss;
                ss << m;
                marker_label.text = ss.str();
            }
                
        }

        marker_label.header.frame_id="/scan";
        marker_label.header.stamp = ros::Time();//to show every tag  (ros::Time::now()for latest tag)
        marker_label.action = visualization_msgs::Marker::ADD;
        marker_label.pose.orientation.w = 1.0;
        marker_label.id = i;
        marker_label.type = visualization_msgs::Marker::TEXT_VIEW_FACING;

        marker_label.scale.z = 1.5f;
        marker_label.color.b = 0.0f;
        marker_label.color.g = 0.9f;
        marker_label.color.r = 0.9f;
        marker_label.color.a = 1;

        marker_label.pose.position.x = temp.center.x;
        marker_label.pose.position.y = temp.center.y;
        marker_label.pose.position.z = temp.center.z + 1.0f;
        
        M_label.markers.push_back(marker_label);
        
    }

    if(M_array.markers.size() > max_size){
        max_size = M_array.markers.size();
    }

    for (int a = i; a < max_size; a++)
    {
        marker.id = a;
        marker.color.a = 0;
        marker.pose.position.x = 0;
        marker.pose.position.y = 0;
        marker.pose.position.z = 0;
        marker.scale.z = 0;
        M_array.markers.push_back(marker);

        marker_label.id = a;
        marker_label.color.a = 0;
        marker_label.pose.position.x = 0;
        marker_label.pose.position.y = 0;
        marker_label.pose.position.z = 0;
        marker_label.scale.z = 0;
        M_label.markers.push_back(marker_label);
    }

    pub_marker.publish(M_array);
    pub_gt.publish(M_label);
    
}


void sort_timestamp(){
	int i=0;

    //sorting lidar
	for(i; i<plys.size(); i++){
		string str = plys.at(i).substr(3, 18);//star at 3th index and read 18 char(timestamp in file name)
				
		istringstream is(str);
		unsigned long int ts;
		is >> ts;
		sorted_plys.push_back(ts);
		//cout << sorted_plys.at(i) <<endl;
	}


    cout << sorted_plys.size() <<endl;
	sort(sorted_plys.begin(), sorted_plys.end());
   
    cout<<"\033[1;33mAfter sorting:\n\033[0m";
	for(i=0;i<sorted_plys.size();i++)
        cout<<sorted_plys[i]<<"\n";

    cout << "get out of function"; 
}


int main(int argc, char** argv){
    
    ros::init(argc,argv,"result_viz");
    ros::NodeHandle nh;

    showAllFiles( dir_lidar, 0 );

    sort_timestamp();

    //test read one last frame
    cout<<"Done read file";
    id_record.resize(0);
     
    pub_scan = nh.advertise<sensor_msgs::PointCloud2>("/scan",1000);
    pub_marker = nh.advertise<visualization_msgs::MarkerArray>("/marker_viz", 1000);
    pub_gt = nh.advertise<visualization_msgs::MarkerArray>("/marker_id", 1000);
    ros::Rate r(rate);
    for(int i=0; i<sorted_plys.size(); i++){
        pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
        sensor_msgs::PointCloud2 sensor_scan;

        stringstream ss;
        ss << sorted_plys.at(i);
        string lidar_time = ss.str();

        string ply_path = data_path + "/train1_6f153f9c-edc5-389f-ac6f-40705c30d97e/lidar/PC_" + ss.str() + ".ply";
        // string label_path = data_path + "/tracking_output/" + log_name + "_lane_1_fixbbox_1_rcnn_1_map_const_v_both_roi/tracked_object_labels_"+ ss.str() + ".json";
        string label_path = data_path + "/tracking_output/train1_6f153f9c-edc5-389f-ac6f-40705c30d97e_lane_1_fixbbox_1_rcnn_1_map_const_v_both_roi/tracked_object_labels_"+ ss.str() + ".json";
        
        if (pcl::io::loadPLYFile<PointT>(ply_path, *cloud) == -1){
            PCL_ERROR("Couldn't read",plys.at(i),"file.\n");
            cout << i << endl;
            continue;
        }
        else{
            Json::Reader reader_label;
            ifstream ifs1(label_path,ios::binary);

            if(!reader_label.parse(ifs1, arrayObj)){
                cout<<"\033[1;34mFailed to parse "<<sorted_plys[i]<<"th label.\033[0m"<<endl;
                continue;
            }
            else{
                num = arrayObj.size();
                cout <<"We have "<< num <<" annotations."<<endl;
                initialize_label();
            }


            M_array.markers.clear();
            M_label.markers.clear();
            draw_box();

            pcl::toROSMsg(*cloud,sensor_scan);
            sensor_scan.header.frame_id = "/scan";
            pub_scan.publish(sensor_scan);

            r.sleep();

        }

    }
   
    
    }
    