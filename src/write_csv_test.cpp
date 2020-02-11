// write_csv.cpp
// updated 8/12/16

// the purpose of this code is to demonstrate how to write data
// to a csv file in C++

// inlcude iostream and string libraries
#include <iostream>
#include <fstream>
#include <string>
#include <ros/ros.h>


using namespace std;

// create an ofstream for the file output (see the link on streams for
// more info)
ofstream outputFile;
ofstream fs;
// create a name for the file output
std::string filename = "exampleOutput.csv";

// create some variables for demonstration
int i;
int A;
int B;
int C;


int main(int argc, char** argv)
{
    ros::init(argc, argv, "test_csv_write");
    ros::NodeHandle nh;
    // create and open the .csv file
    outputFile.open(filename);
    
    // write the file headers
    outputFile << "Column A" << "," << "Column B" << "Column C" << std::endl;
    
    // i is just a variable to make numbers with in the file
    i = 1;
  
    // write data to the file
    for (int counter = 0; counter <  10; counter++)
    {
        // make some data
        A = i + 5;
        B = i + 10;
        C = i + 20;
        
        // write the data to the output file
        outputFile << A << "," << B << "," << C << std::endl;
        cout << A << "," << B << "," << C << std::endl;
        
        // increase i
        i = i * 5;
    }
    
    // close the output file
    outputFile.close();

    // outputFile.open(filename);
    // outputFile <<"TEST"<<std::endl;
    // outputFile.close();
    return 0;   
}
