#include "core.hpp";
#include "highgui.hpp"
#include "imgcodecs.hpp"

using namespace cv;
using namespace std;

int main()
{
Mat img;
img = imread("pic.jpg");
imshow("Original Image", img);
waitKey();
}
