////publish lidar pt and annotations , lidar frame index to local_driveable_map_sub.py to get driveable area
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <json/json.h>
#include <ros/ros.h>
#include <fstream>
#include <vector>


#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>
#include <std_msgs/Int16.h>
#include <std_msgs/Time.h>
#include <std_msgs/Header.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

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
#include <ros/package.h>
//read all .ply in files
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <algorithm>
#include <tf/LinearMath/Matrix3x3.h>
#include <tf/LinearMath/Vector3.h>
#include <tf/transform_broadcaster.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/image_encodings.h>
using namespace std;
typedef pcl::PointXYZI PointT;
#define de 0.5//2   0.5 for good recording at 2x
#define rate 2//5  5  

typedef struct an{
    geometry_msgs::Point center;
    geometry_msgs::Quaternion rotation;
    double length;
    double width;
    double height;
    string track_id;
    double time_s;
    string track_class;
 }an;
char dir_lidar[100] = "/home/ee904/ARGO/argoverse-tracking/train4/91326240-9132-9132-9132-591327440896/lidar"; //read lidar ply filename
/////////////////////////////////////modifiying lidar index queue to 200
char dir_stereo_l [200] = "/home/ee904/ARGO/argoverse-tracking/train4/91326240-9132-9132-9132-591327440896/stereo_front_left";
char dir_stereo_r[200]="/home/ee904/ARGO/argoverse-tracking/train4/91326240-9132-9132-9132-591327440896/stereo_front_right";


string data_path = "/home/ee904/ARGO/argoverse-tracking/train4/91326240-9132-9132-9132-591327440896";

int const MAX_STR_LEN = 200;
vector <string> plys,stereos_l,stereos_r;
//vector <string> labels;
vector <unsigned long int> sorted_plys,sorted_stereo_left,sorted_stereo_right;
//vector <unsigned long int> sorted_labels;
vector <pcl::PointCloud<PointT>::Ptr> clouds;
sensor_msgs::PointCloud2 sensor_scan;
std_msgs::Int16 frame_index;
std_msgs::Time t;
sensor_msgs::Image stereo_left,stereo_right;


vector <an> annotations;
Json::Value arrayObj, se3Obj;
visualization_msgs::MarkerArray M_array;
int max_size=0;
int num;
int count_stereo=0;

//const vector <string> category = {" ","VEHICLE","PEDESTRIAN","ON_ROAD_OBSTACLE","LARGE_VEHICLE","BICYCLE","BICYCLIST","BUS","OTHER_MOVER","TRAILER","MOTOCYCLIST","MOPED","MOTORCYCLE","STROLLER","EMERGENCY_VEHICLE","ANIMAL","WHEELCHAIR","SCHOOL_BUS"};
const vector <string> category = {" ","SCHOOL_BUS","WHEELCHAIR","ANIMAL","EMERGENCY_VEHICLE","STROLLER","MOTORCYCLE","MOPED","MOTOCYCLIST","TRAILER","OTHER_MOVER","BUS","BICYCLIST","BICYCLE","LARGE_VEHICLE","ON_ROAD_OBSTACLE","PEDESTRIAN","VEHICLE"};
#define indent 0.05


ros::Publisher pub_scan;
ros::Publisher pub_marker;
ros::Publisher pub_index;
ros::Publisher pub_time;
ros::Publisher pub_stereo_l,pub_stereo_r;


//vector <tf::Matrix3x3 rotation_matrix> rotation;
tf::Transform scan_transform ;

//void showAllFiles( const string dir_name, int shit )
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
        else if ( shit==1 )
            stereos_l.push_back(filename->d_name);
        else
            stereos_r.push_back(filename->d_name);
        
    }   
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

    cout << stereos_l.size() << stereos_r.size() <<endl;

    for(i=0; i<stereos_l.size(); i++){
		string str_l = stereos_l.at(i).substr(18, 18);//star at 3th index and read 18 char(timestamp in file name)
		string str_r = stereos_r.at(i).substr(19, 18);		
		istringstream is_l(str_l);
        istringstream is_r(str_r);
		unsigned long int s_l,s_r;
		is_l >> s_l;
        is_r >> s_r;
		sorted_stereo_left.push_back(s_l);
        sorted_stereo_right.push_back(s_r);
		//cout << "\033[1;33mStereo_l at " <<i<< " :" <<sorted_stereo_left.at(i) <<"\033[0m"<<endl;
        //cout << "Stereo_r at " <<i<< " :" <<sorted_stereo_right.at(i) <<endl;
	}

    //sorting label
	// for(i; i<labels.size(); i++){
	// 	string str = labels.at(i).substr(28, 12);
			
	// 	istringstream is(str);
	// 	unsigned long int ts;
	// 	is >> ts;
	// 	sorted_labels.push_back(ts);
	// 	cout << sorted_labels.at(i) <<endl;
	// }
    cout << sorted_plys.size() <<"\n"<< sorted_stereo_left.size() << "\n" << sorted_stereo_right.size() <<endl;
	sort(sorted_plys.begin(), sorted_plys.end());
    sort(sorted_stereo_left.begin(), sorted_stereo_left.end());
    sort(sorted_stereo_right.begin(), sorted_stereo_right.end());
    //sort(sorted_labels.begin(), sorted_labels.end());
    cout<<"\033[1;33mAfter sorting:\n\033[0m";
	for(i=0;i<sorted_plys.size();i++)
        cout<<sorted_plys[i]<<"\n";

    cout<<"Stereo right"<<endl;
    for(i=0;i<sorted_stereo_right.size();i++)
        cout<<sorted_stereo_right.at(i)<<"\n";
    cout << "get out of function"; 
}

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
        temp.track_id = arrayObj[i]["track_label_uudi"].asString();
        temp.track_class = arrayObj[i]["label_class"].asString();
        annotations.push_back(temp);
    }
    cout<<"Done initialize."<<endl;
}



void draw_box(ros::Time lidar_timestamp){
    visualization_msgs::Marker marker;
    int i;
    for(i=0; i<num; i++){
        an temp = annotations.at(i);
        marker.header.frame_id="/scan";
        //marker.header.stamp = ros::Time();//to show every tag  (ros::Time::now()for latest tag)
        marker.header.stamp = lidar_timestamp;
        marker.ns = temp.track_class;
        
        marker.action = visualization_msgs::Marker::ADD;
        marker.lifetime = ros::Duration(de);
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
        marker.color.r = 0.0f;
	    marker.color.a = 0.7f;

        // int k;
        // for(k=1; k<category.size(); k++){
        //     if( temp.track_class.compare(category.at(k)) == 0 ){
		//         //cout << category.at(k) <<endl;
        //         int color = k % 3;
        //         if ( color == 0 ){
        //             marker.color.b = 0.0f + k*indent;
		//     //cout << "marker.color.b" << marker.color.b <<endl;
		// }
        //         else if ( color == 1 ){
        //             marker.color.g = 0.0f + k*indent;
		//     //cout << "marker.color.g" << marker.color.g <<endl;
		// }                
		// else{ 
        //             marker.color.r = 0.0f + k*indent;
		//     //cout << "marker.color.r" << marker.color.r <<endl;
		// }
	    // }
                
        // }
        //seting color
        float INDENT = (float)1/256;
        if( temp.track_class == "VEHICLE")
            marker.color.r = 1.0f;
        else if( temp.track_class == "LARGE_VEHICLE")
            marker.color.r = INDENT*205;
        else if( temp.track_class == "BUS")
            marker.color.r = INDENT*238;
        else if( temp.track_class == "EMERGENCY_VEHICLE")
            marker.color.r = INDENT*139;
        else if( temp.track_class == "SCHOOL_BUS")
            marker.color.r = INDENT*100;
        else if( temp.track_class =="TRAILER")
            marker.color.r = INDENT*70;
            
        else if (temp.track_class == "PEDESTRIAN")
            marker.color.g = 1.0f;

        else if( temp.track_class == "BICYCLE")
            marker.color.b = 1.0f;
        else if( temp.track_class =="MOTOCYCLE")
            marker.color.b = INDENT*238; 
        else if( temp.track_class == "MOPED")
            marker.color.b = INDENT*139;
        
        else if( temp.track_class == "BICYCLIST"){
            marker.color.r = INDENT*194;
            marker.color.g = INDENT*228;
            marker.color.b = INDENT*185;
        }
        else if( temp.track_class == "MOTORCYCLIST"){
            marker.color.r = INDENT*127;
            marker.color.g = INDENT*255;
        }

        else if( temp.track_class == "ON_ROAD_OBSTACLE"){
            marker.color.r = INDENT*255;
            marker.color.g = INDENT*77;
            marker.color.b = INDENT*225;
        }

        else if( temp.track_class == "STROLLER"){
            marker.color.r = INDENT*252;
            marker.color.g = INDENT*185;
            marker.color.b = INDENT*29;
        }
        else if( temp.track_class == "WHEELCHAIR"){
            marker.color.r = INDENT*255;
            marker.color.g = INDENT*165;
        }
        //OTHER_MOVER/ANIMAL
        else if(temp.track_class == "OTHER_MOVER"){
            marker.color.r = INDENT*160;
            marker.color.g = INDENT*32;
            marker.color.b = INDENT*240;
        }
        else{
            marker.color.r = INDENT*218;
            marker.color.g = INDENT*112;
            marker.color.b = INDENT*214;
        }

        M_array.markers.push_back(marker);
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
    }

    pub_marker.publish(M_array);
    
}




int main(int argc, char** argv){
    
    ros::init(argc,argv,"draw_box");
    ros::NodeHandle nh;

    showAllFiles( dir_lidar, 0 );
    //showAllFiles( dir_label, true );

    showAllFiles( dir_stereo_l, 1 );
    showAllFiles( dir_stereo_r, 2 );
    sort_timestamp();

    //test read one last frame
    cout<<"Done read file";
    // image_transport::ImageTransport it(nh);
    // image_transport::Publisher pub_stereo_l = it.advertise("/stereo_img_left");
    // image_transport::Publisher pub_stereo_r = it.advertise("/stereo_img_right");
    cv_bridge::CvImage cv_image_l,cv_image_r;
    pub_marker = nh.advertise<visualization_msgs::MarkerArray>("/anno_marker", 1000);
    pub_scan = nh.advertise<sensor_msgs::PointCloud2>("/scan",1000);
    pub_index = nh.advertise<std_msgs::Int16>("/lidar_index",300); //queue size 1 for sync, 300 for rosbag##########
    // pub_time = nh.advertise<std_msgs::Time>("/time",300);
    pub_stereo_l = nh.advertise<sensor_msgs::Image>("/stereo_img_left",1);
    pub_stereo_r = nh.advertise<sensor_msgs::Image>("/stereo_img_right",1);

    //string pkg_path = ros::package::getPath("point_cloud_io");

    int i=0;
	
    ros::Rate r(rate);
    //while (ros::ok()){ //&& done !=true(or count!= logs#)
    for (i; i<sorted_plys.size(); i++){
        cout << "Finish" << data_path << endl;
        pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
        stringstream ss;
        ss << sorted_plys.at(i);
        string lidar_time = ss.str();

        string ply_path = data_path + "/lidar/PC_" + ss.str() + ".ply";
        string label_path = data_path + "/per_sweep_annotations_amodal/tracked_object_labels_" + ss.str() + ".json";
         
        string se3_path = data_path + "/poses/city_SE3_egovehicle_" + ss.str() +".json";
        string s_l;
        string s_r;
        
        /////////////for stereo
        unsigned long int ste_l_time, ste_r_time;
        stringstream ste_l,ste_r;
        ste_l << sorted_stereo_left.at(count_stereo);
        ste_r << sorted_stereo_right.at(count_stereo);
        ste_l_time = sorted_stereo_left.at(count_stereo);
        ste_r_time = sorted_stereo_right.at(count_stereo);
        string ste_time = ste_l.str() ;
        if (count_stereo == 0 || ste_time.substr(0,10) == lidar_time.substr(0,10)){
            s_l = data_path + "/stereo_front_left/stereo_front_left_" + ste_l.str() + ".jpg";
            s_r = data_path + "/stereo_front_right/stereo_front_right_" + ste_r.str() + ".jpg";
            count_stereo++;
            
            
            cv_image_l.image = cv::imread(s_l, CV_LOAD_IMAGE_COLOR);
            cv_image_l.encoding = "bgr8";
            //cv_image_l.header.stamp = current_time;
            cv_image_l.header.stamp.sec = ste_l_time  / 1000000000 + 1000000000;
            cv_image_l.header.stamp.nsec = ste_l_time % 1000000000;
            cout << cv_image_l.header.stamp << endl;

            cv_image_l.toImageMsg(stereo_left);
            
            cv_image_r.image = cv::imread(s_r, CV_LOAD_IMAGE_COLOR);
            cv_image_r.encoding = "bgr8";
            //cv_image_r.header.stamp = current_time;
            cv_image_r.header.stamp.sec = ste_r_time / 1000000000 + 1000000000;
            cv_image_r.header.stamp.nsec = ste_r_time % 1000000000;

            cv_image_r.toImageMsg(stereo_right);
            
            pub_stereo_l.publish(stereo_left);
            pub_stereo_r.publish(stereo_right);
            cout<<"Publishing stereo."<<endl;

        }
        // else{
        //     cout <<"The same stereo @" << i << endl;
        //     if (count_stereo != 0){
        //         ste_l << sorted_stereo_left.at(count_stereo-1);
        //         ste_r << sorted_stereo_right.at(count_stereo-1);
        //         // ste_l_time = sorted_stereo_left.at(count_stereo);
        //         // ste_r_time = sorted_stereo_right.at(count_stereo);
        //         ste_l_time = sorted_stereo_left.at(count_stereo-1);
        //         ste_r_time = sorted_stereo_right.at(count_stereo-1);
        //         s_l = data_path + "/stereo_front_left/stereo_front_left_" + ste_l.str() + ".jpg";
        //         s_r = data_path + "/stereo_front_right/stereo_front_right_" + ste_r.str() + ".jpg";
        //     }
        //     else
        //     {
        //         s_l = data_path + "/stereo_front_left/stereo_front_left_" + ste_l.str() + ".jpg";
        //         s_r = data_path + "/stereo_front_right/stereo_front_right_" + ste_r.str() + ".jpg";
        //         count_stereo++;
        //     }
            

        // }
      


        if (pcl::io::loadPLYFile<PointT>(ply_path, *cloud) == -1){
            PCL_ERROR("Couldn't read",plys.at(i),"file.\n");
            cout << i << endl;
            continue;
        }
        else{
            Json::Reader reader_label, reader_se3;
            ifstream ifs1(label_path,ios::binary);
            ifstream ifs2(se3_path,ios::binary);
            
            if(!reader_label.parse(ifs1, arrayObj)){
                cout<<"\033[1;34mFailed to parse "<<sorted_plys[i]<<"th label.\033[0m"<<endl;
                continue;
            }
            else{
                num = arrayObj.size();
                cout <<"We have "<< num <<" annotations."<<endl;
                initialize_label();
                        
            }

            if(!reader_se3.parse(ifs2, se3Obj)){
                cout<<"\033[1;34mFailed to parse "<<sorted_plys[i]<<"th se3.\033[0m"<<endl;
                continue;
            }
            else{
                int num1 = se3Obj.size();
                cout <<"We have "<< num1 <<" fields."<<endl;
            }
        


            // ros::Time current_time = ros::Time::now();
            // t.data = current_time;
            // pub_time.publish(t);
            ros::Time lidar_timestamp ;
            lidar_timestamp.sec = sorted_plys[i] / 1000000000 + 1000000000;
            lidar_timestamp.nsec = sorted_plys[i] % 1000000000;

            tf::Vector3 translation_matrix;
            //w is in first field in pose.json (w,x,y,z)  and (x,y,z,w) for tf
            tf::Quaternion tfqt = tf::Quaternion( se3Obj["rotation"][1].asDouble(), se3Obj["rotation"][2].asDouble(), se3Obj["rotation"][3].asDouble(), se3Obj["rotation"][0].asDouble());
            translation_matrix.setValue(se3Obj["translation"][0].asDouble(),se3Obj["translation"][1].asDouble(),se3Obj["translation"][2].asDouble());
            scan_transform.setOrigin(translation_matrix);
            scan_transform.setRotation(tfqt);
            //cout << tfqt;
            //cout << translation_matrix;
            cout << se3Obj["translation"][0].asDouble() << "," << se3Obj["translation"][1].asDouble() << ","<< se3Obj["translation"][2].asDouble()<<endl;

            static tf::TransformBroadcaster br;
            //br.sendTransform(tf::StampedTransform(scan_transform , ros::Time::now(), "map", "scan"));
            //br.sendTransform(tf::StampedTransform(scan_transform , current_time, "map", "scan"));
            br.sendTransform(tf::StampedTransform(scan_transform , lidar_timestamp, "map", "scan"));

            M_array.markers.clear();
            //draw_box(current_time);
            draw_box(lidar_timestamp);
            
            frame_index.data = i;
            pub_index.publish(frame_index);
                                    
            clouds.push_back(cloud);
            pcl::toROSMsg(*cloud,sensor_scan);
            sensor_scan.header.frame_id = "/scan";
            //sensor_scan.header.stamp = current_time;
            sensor_scan.header.stamp = lidar_timestamp;
            pub_scan.publish(sensor_scan);
            cout << "\033[1;33m@" <<i<<" "<< sensor_scan.header.stamp <<"\033[0m"<< endl;
            //publish to get local_driveable_area
           
            // cv_image_l.image = cv::imread(s_l, CV_LOAD_IMAGE_COLOR);
            // cv_image_l.encoding = "bgr8";
            // //cv_image_l.header.stamp = current_time;
            // cv_image_l.header.stamp.sec = ste_l_time  / 1000000000 + 1000000000;
            // cv_image_l.header.stamp.nsec = ste_l_time % 1000000000;
            // cout << cv_image_l.header.stamp << endl;

            // cv_image_l.toImageMsg(stereo_left);
            
            // cv_image_r.image = cv::imread(s_r, CV_LOAD_IMAGE_COLOR);
            // cv_image_r.encoding = "bgr8";
            // //cv_image_r.header.stamp = current_time;
            // cv_image_r.header.stamp.sec = ste_r_time / 1000000000 + 1000000000;
            // cv_image_r.header.stamp.nsec = ste_r_time % 1000000000;

            // cv_image_r.toImageMsg(stereo_right);
            
            // pub_stereo_l.publish(stereo_left);
            // pub_stereo_r.publish(stereo_right);
            // cout<<"Publishing stereo."<<endl;


            r.sleep();

        }
    }
    

    //}
    // ros::Rate r(10);
    // while(ros::ok()){
    //    pub_marker.publish(M_array);
    //    pub_scan.publish(sensor_scan);
    //    pub_index.publish(frame_index);
    //    ros::spinOnce();
    //    r.sleep();
        
    //}

    cout << "Finish" << data_path << endl;
    return 0;
}
