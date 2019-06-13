/**

	dropper
	main.cpp
	Obtains image from front camera, edits it, and puts it on a file containing error (log) and matrix (imgP)

	@author Luke Shimanuki
	@version 1.0

*/


#include <vector>
#include <functional>
#include <algorithm>
#include <thread>
#include <iostream>
#include <chrono>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.h>

#define SIGNAL 123456.78910

double perp_slope(cv::Vec4i v) {
	if (v[0] - v[2] == 0) {
		return 0;
	}
	if (v[1] - v[3] == 0) {
		return SIGNAL;
	}
	return -pow(((v[1] - v[3]) / (v[0] - v[2])), -1);
}

cv::Vec2i midpoint(cv::Vec4i v) {
	cv::Vec2i doobly;
	doobly[0] = (int) ((v[0] + v[2]) / 2);
	doobly[1] = (int) ((v[1] + v[3]) / 2);
	return doobly;
}

cv::Vec4i long_line(cv::Vec4i v) {
	double slope = perp_slope(v);
	cv::Vec2i point = midpoint(v);
	cv::Vec4i r;

	if (slope == SIGNAL) {
		r[0] = point[0];
		r[2] = point[0];
		r[1] = 0;
		r[3] = 645;
		return r;
	}

	double y_int = -(slope * point[0]) + point[1];

	r[0] = -1;
	r[1] = -1 * slope + y_int;
	r[2] = 645;
	r[3] = 645 * slope + y_int;
	return r;
}

cv::Vec2i intersection(cv::Vec4i v1, cv::Vec4i v2) {
	int x1 = v1[0];		
	int y1 = v1[1];		
	int x2 = v1[2];		
	int y2 = v1[3];		

	int x3 = v2[0];		
	int y3 = v2[1];		
	int x4 = v2[2];		
	int y4 = v2[3];		

	double den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

	double a1 = perp_slope(v1);
	double a2 = perp_slope(v2);
	a1 = (a1 == SIGNAL) ? 1 : atan(a1);
	a2 = (a2 == SIGNAL) ? 1 : atan(a2);

	if (den == 0 || abs(a1 - a2) > (M_PI / 2.75f)) {
		return cv::Vec2i(-1, -1);
	}

	double top1 = ((x1 * y2 - y1 * x2) * (x3 - x4)) - ((x1 - x2) * (x3 * y4 - y3 * x4));
	double top2 = ((x1 * y2 - y1 * x2) * (y3 - y4)) - ((y1 - y2) * (x3 * y4 - y3 * x4));

	return cv::Vec2i((int) (top1 / den), (int) (top2 / den));
}

int main(int argc, char** argv)
{
	FILE* in = stdin;
	FILE* out = stdout;
	FILE* log = stderr;

	if(argc < 2) {
		std::cout << "Incorrect usage: '$ ./dropper <filepath>'";
	}

	cv::Mat mask = cv::imread("mask.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

	cv::Mat masked;
	cv::addWeighted(mask, 0, image, 1, 0, masked);

	cv::Mat denoised;
    cv::GaussianBlur(masked, denoised, cv::Size(5,5), 0, 0, cv::BORDER_DEFAULT );

	cv::Mat laplace;
	int kernel_size = 3;
	int scale = 1;
	int delta = 0;

	cv::Laplacian(denoised, laplace, CV_16S, kernel_size, scale, delta, cv::BORDER_DEFAULT);
	cv::convertScaleAbs(laplace, laplace);
	cv::Mat collap;
	int thresh_type = 1; // invert binary
    cv::threshold(laplace, laplace, 1, 255, thresh_type);
	// int thresh_type = 0; //binary
    // cv::threshold(laplace, laplace, 10, 255, thresh_type);
	cv::Mat opened;

	int erosion_size = 1;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1), cv::Point(erosion_size, erosion_size));
	cv::erode(laplace, opened, element);
	erosion_size = 4;
	element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1), cv::Point(erosion_size, erosion_size));
	cv::dilate(opened, opened, element);

	thresh_type = 0; // binary
	cv::Mat mask2;
    cv::threshold(image, mask2, 127, 255, thresh_type);
	cv::Mat masked2;
	opened.copyTo(masked2, mask2);

	int dilation_size = 4;
	element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1), cv::Point(dilation_size, dilation_size));
	//cv::dilate(masked2, masked2, element);

	cv::Mat can, colcan;
	cv::Canny(masked2, can, 50, 200, 3);
	cv::cvtColor(can, colcan, CV_GRAY2BGRA);

	// Hough line detection
	std::vector<cv::Vec4i> lines;
	std::vector<cv::Vec4i> plines;
	int ytotal = 0;
	int xtotal = 0;
	int ptotal = 0;
	cv::HoughLinesP(can, lines, 1, CV_PI/180, 30, 0, 0);
	cv::Mat lineim = cv::Mat::zeros(colcan.rows, colcan.cols, colcan.type());
	for(size_t i = 0; i < lines.size(); i++) {
		cv::Vec4i p = long_line(lines[i]);
		plines.push_back(p);
	}

	for(size_t i = 0; i < plines.size(); i++) {
		for(size_t j = 0; j < plines.size(); j++) {
			if(i == j) {
				continue;
			}
			cv::Vec2i inter = intersection(plines[i], plines[j]);
			int x = inter[0];
			int y = inter[1];
			if(x > 0 && x < colcan.cols && y > 0 && y < colcan.rows) {
				cv::circle(colcan, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), 3);
				xtotal += x;
				ytotal += y;
				ptotal++;
			}
		}
	}
		

	cv::Mat color;
	cv::cvtColor(image, color, CV_GRAY2BGR);

	if (ptotal > 0) {
		int ycentroid = (int) ytotal / ptotal;
		int xcentroid = (int) xtotal / ptotal;
		std::cout << xcentroid << "," << ycentroid << std::endl;
		cv::circle(color, cv::Point(xcentroid, ycentroid), 10, cv::Scalar(0, 255, 0), 3);
	}

	cv::imshow("final", colcan);
	cv::imshow("src", color);
	cv::waitKey();
}

