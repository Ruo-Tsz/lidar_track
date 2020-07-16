#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <string>
#include <ros/package.h>

int counter = 0;
std::string pkg_path = ros::package::getPath("lidar_track"); 

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  ros::Time timestamp = msg->header.stamp;
  std::cout << timestamp << std::endl;
  double stamp = timestamp.sec * 1000000000.0 + timestamp.nsec;
  std::cout << stamp <<std::endl;
  std::stringstream ss_stamp;
  ss_stamp << std::setprecision(19) << stamp << std::endl;
  // std::string output_path = pkg_path + "/temp/" + ss_stamp.str() + ".jpeg";
  std::string output_path = pkg_path + "/temp/" + ss_stamp.str() + ".png";
  std::cout<< output_path << std::endl;
  try
  {
    cv_bridge::CvImageConstPtr cv_ptr (new cv_bridge::CvImage);
    cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    std::cout << cv_ptr -> image.rows << " " <<cv_ptr -> image.cols;

    // show and save as .png
    cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
    cv::waitKey(30);
    cv::imwrite( output_path, cv_bridge::toCvShare(msg, "bgr8")->image );
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "image_listener");
  ros::NodeHandle nh;
  cv::namedWindow("view");
  cv::startWindowThread();
  image_transport::ImageTransport it(nh);
  // get raw image from republished raw topic by image_transport package
  // rosrun image_tlish compressed in:=/pylon_camera_node/image_raw out:=/pylon_camera_node/repub
  image_transport::Subscriber sub = it.subscribe("/pylon_camera_node/repub", 1, imageCallback);

  ros::spin();
  cv::destroyWindow("view");
}