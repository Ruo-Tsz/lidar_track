#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <string>
#include <iostream>
#include <fstream>
using namespace std;
namespace enc = sensor_msgs::image_encodings;
image_transport::Publisher pub;

void internalCallback(const sensor_msgs::CompressedImageConstPtr& message){

  cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);
  // cout << "in callback" << endl;

  // Copy message header
  cv_ptr->header = message->header;

  // Decode color/mono image
  try
  {
    cv_ptr->image = cv::imdecode(cv::Mat(message->data), cv::IMREAD_COLOR);

    // Assign image encoding string
    const size_t split_pos = message->format.find(';');
    if (split_pos==std::string::npos)
    {
      // Older version of compressed_image_transport does not signal image format
      switch (cv_ptr->image.channels())
      {
        case 1:
          cv_ptr->encoding = enc::MONO8;
          break;
        case 3:
          cv_ptr->encoding = enc::BGR8;
          break;
        default:
          ROS_ERROR("Unsupported number of channels: %i", cv_ptr->image.channels());
          break;
      }
    } else
    {
      std::string image_encoding = message->format.substr(0, split_pos);

      cv_ptr->encoding = image_encoding;
      cout << "encoder:" << image_encoding << endl;
      if ( enc::isColor(image_encoding))
      {
        std::string compressed_encoding = message->format.substr(split_pos);
        bool compressed_bgr_image = (compressed_encoding.find("compressed bgr") != std::string::npos);

        // Revert color transformation
        if (compressed_bgr_image)
        {
          // if necessary convert colors from bgr to rgb
          if ((image_encoding == enc::RGB8) || (image_encoding == enc::RGB16))
            cv::cvtColor(cv_ptr->image, cv_ptr->image, CV_BGR2RGB);

          if ((image_encoding == enc::RGBA8) || (image_encoding == enc::RGBA16))
            cv::cvtColor(cv_ptr->image, cv_ptr->image, CV_BGR2RGBA);

          if ((image_encoding == enc::BGRA8) || (image_encoding == enc::BGRA16))
            cv::cvtColor(cv_ptr->image, cv_ptr->image, CV_BGR2BGRA);
        } else
        {
          // if necessary convert colors from rgb to bgr
          if ((image_encoding == enc::BGR8) || (image_encoding == enc::BGR16))
            cv::cvtColor(cv_ptr->image, cv_ptr->image, CV_RGB2BGR);

          if ((image_encoding == enc::BGRA8) || (image_encoding == enc::BGRA16))
            cv::cvtColor(cv_ptr->image, cv_ptr->image, CV_RGB2BGRA);

          if ((image_encoding == enc::RGBA8) || (image_encoding == enc::RGBA16))
            cv::cvtColor(cv_ptr->image, cv_ptr->image, CV_RGB2RGBA);
        }
      }
    }
  }
  catch (cv::Exception& e)
  {
    ROS_ERROR("%s", e.what());
  }

  size_t rows = cv_ptr->image.rows;
  size_t cols = cv_ptr->image.cols;
  cv::imshow("row",cv_ptr->image);
  cv::waitKey(30);
  cout << "row:" << rows << ", col:" <<cols <<endl;
  if ((rows > 0) && (cols > 0)){
    // Publish message to user callback
    //user_cb(cv_ptr->toImageMsg());
    sensor_msgs::ImageConstPtr msg = cv_ptr->toImageMsg();  //get the raw image data
    // cout << msg->data;
  
    pub.publish(*msg);
  }
}
int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_sub");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    ////////subscribe////////
    ros::Subscriber sub_comp = nh.subscribe("/pylon_camera_node/image_raw/compressed", 1, internalCallback);
    pub = it.advertise("/pylon_camera/republish", 1);

    while(ros::ok()){
      ros::spin();
    }

}