/*
  Malhar Mahant & Kruthika Gangaraju
  SP23

  Defines functions to generate different kinds of features from an image and find similar images using generated features.
*/
#pragma once
#ifndef _MATCHFUNCTIONS_H_
#define _MATCHFUNCTIONS_H_
#include <opencv2/opencv.hpp>
namespace matchfunctions {
	/*
	* Helper method to find contours of objects in the binary image
	*/
	int findAllContours(cv::Mat& src, std::vector<std::vector<Point>>& contours, std::vector<Vec4i>& hierarchy);

	/*
	* Helper method to generate features to be used for classification of objects
	*/
	int generateFeatures(std::vector<Point>& contour, std::vector<float>& features);

	/*
	* Generates and saves the features for a given contour for a given label in the training data.
	*/
	int generateAndSaveFeatures(std::string label, std::vector<Point>& contour);

	/*
	* Classify new objects in image using Nearest Neighbor Classification.
	*/
	int nearestNeighbor(std::vector<std::vector<Point>>& contours, std::vector<Vec4i>& hierarchy, std::vector<std::string>& outLabels);

	/*
	* Classify new objects in image using Multi-Class K Nearest Neighbor Classification.
	*/
	int kNearestNeighbor(std::vector<std::vector<Point>>& contours, std::vector<Vec4i>& hierarchy, std::vector<std::string>& outLabels, std::vector <std::string>& sumDistances);
}
#endif
