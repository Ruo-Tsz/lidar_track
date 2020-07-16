#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <string>
#include <iostream>
#include <fstream>
#include <ros/package.h>
using namespace std;
bool save = false;

void internalCallback(const sensor_msgs::CompressedImageConstPtr& message){
    ros::Time timestamp = message->header.stamp;
    cout << timestamp.nsec << endl;
    cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);
    try{
        cv_ptr = cv_bridge::toCvCopy(message, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    std::stringstream sstream_nsce, sstream_sec;
    // unsigned long int stamp = timestamp.sec * 1000000000 + timestamp.nsec;
    sstream_nsce << timestamp.nsec << std::endl;
    sstream_sec << timestamp.sec << std::endl;
    string pkg_path = ros::package::getPath("lidar_track"); 
    cout << sstream_sec.str();
    string out_path = pkg_path + "/" + sstream_sec.str() + sstream_nsce.str() + ".jpeg";
    cout << out_path << endl;
    
    if (!save)
    {
        try{
            cv::imwrite( out_path,  cv_ptr->image );
            ROS_INFO("Save image at timestamp %s", timestamp.nsec);
            save = true;
        }
        catch(cv::Exception& e){
            ROS_ERROR("Saving exception: %s", e.what());
        }
    }

}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_difference");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    ////////subscribe////////
    ros::Subscriber sub_comp = nh.subscribe("/pylon_camera_node/image_raw/compressed", 1, internalCallback);
    // image_transport::Subscriber sub_comp = it.subscribe("/pylon_camera_node/image_raw", 1, internalCallback);
    
    while(ros::ok()){
        ros::spin();
    }

}