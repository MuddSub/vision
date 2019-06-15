#include <iostream>
#include <math.h>
#include <opencv.hpp>
#include <core.hpp>

//#include "opencv2\\opencv.hpp"
//#include "opencv2\\core.hpp"
//#include "opencv2\\highgui.hpp"
//#include "opencv2\\imgproc\\imgproc.hpp"
//#include "opencv2\\imgcodecs\\imgcodecs.hpp"
//g++ gate.cpp -o gate.exe
//g++ -std=c++14 gate.cpp -I C:\OpenCV\opencv\build\include -L C:\OpenCV\opencv\build\include\imgcodecs -L C:\OpenCV\opencv\build\include\imgproc -o gate.exe
using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
 // Read the image file
  Mat image = imread("C:\\Users\\rongk\\Downloads\\test.jpg");

  if (image.empty()) // Check for failure
  {
   cout << "Could not open or find the image" << endl;
   system("pause"); //wait for any key press
   return -1;
  }

  String windowName = "My HelloWorld Window"; //Name of the window

  namedWindow(windowName); // Create a window

  imshow(windowName, image); // Show our image inside the created window.

  waitKey(0); // Wait for any keystroke in the window

  destroyWindow(windowName); //destroy the created window

  return 0;
}
