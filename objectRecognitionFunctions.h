#pragma once
/*
  Malhar Mahant & Kruthika Gangaraju
  SP23

  Utility functions for object recognition.
 */
#ifndef _OBJECTRECOGNITIONFUNCTIONS_H_
#define _OBJECTRECOGNITIONFUNCTIONS_H_
#include <opencv2/opencv.hpp>
using namespace cv;

namespace objectrecognition {
	/*
	* Helper method to generate a binary image given a greyscale image
	*/
	int generateBinaryImage(cv::Mat& src, cv::Mat& dst, int threshold);

	/*
	* Helper method to run erosion on a given binary image with given connectedness metric and number of iterations.
	*/
	int erosion(cv::Mat& src, cv::Mat& dst, int connectedness, int iterations);

	/*
	* Helper method to run dilation on a given binary image with given connectedness metric and number of iterations.
	*/
	int dilation(cv::Mat& src, cv::Mat& dst, int connectedness, int iterations);

	/*
	* Helper method that implements grassfire transform to fill holes in the image.
	*/
	int grassfireTransform(cv::Mat& src, cv::Mat& dst);

	/*
	* Helper function to segment the image into regions.
	*/
	int segmentImage(cv::Mat& src, cv::Mat& labels, cv::Mat& stats, cv::Mat& centroids, std::vector<Vec3b>& colors);

	/*
	* Helper function to highlight only the selected region.
	*/
	int selectRegion(cv::Mat& src, cv::Mat& dst, int numlabels, int selected, cv::Mat& labels, cv::Mat& stats, std::vector<Vec3b>& colors);
}
#endif