/*
Malhar Mahant & Kruthika Gangaraju
SP23

Implements different object recognition functions.
*/
#include "objectRecognitionFunctions.h"

// Minimum area of pixels to consider a valid object
int minNumPixels = 2048;

/*
* Helper method to generate a binary image given a greyscale image
*/
int objectrecognition::generateBinaryImage(cv::Mat& src, cv::Mat& dst, int threshold)
{
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if ((short)src.at<uchar>(i, j) < threshold) // Try changing to 128
			{
				dst.at<uchar>(i, j) = 255;
			}
			else
			{
				dst.at<uchar>(i, j) = 0;
			}
		}
	}
	return 0;
}

/*
* Helper method to run erosion on a given binary image with given connectedness metric and number of iterations.
*/
int objectrecognition::erosion(cv::Mat& src, cv::Mat& dst, int connectedness, int iterations)
{
	src.copyTo(dst);
	while (iterations != 0) {
		for (int i = 1; i < src.rows - 1; i++) {
			for (int j = 1; j < src.cols - 1; j++) {
				if (i == 0 || i == src.rows - 1 || j == 0 || j == src.cols - 1) {
					dst.at<uchar>(i, j) = 0;
					continue;
				}
				if (connectedness == 8) {
					if ((short)src.at<uchar>(i - 1, j - 1) == 0 || (short)src.at<uchar>(i - 1, j) == 0 || (short)src.at<uchar>(i - 1, j + 1) == 0 || (short)src.at<uchar>(i, j - 1) == 0 || (short)src.at<uchar>(i, j + 1)
						== 0 || (short)src.at<uchar>(i + 1, j - 1) == 0 || (short)src.at<uchar>(i + 1, j) == 0 || (short)src.at<uchar>(i + 1, j + 1) == 0) {
						dst.at<uchar>(i, j) = 0;
					}
				}
				else {
					if ((short)src.at<uchar>(i - 1, j) == 0 || (short)src.at<uchar>(i, j - 1) == 0 || (short)src.at<uchar>(i, j + 1)
						== 0 || (short)src.at<uchar>(i + 1, j) == 0) {
						dst.at<uchar>(i, j) = 0;
					}/*
					else {
						dst.at<uchar>(i, j) = 255;
					}*/
				}
			}
		}
		iterations--;
	}
	return 0;
}

/*
* Helper method to run dilation on a given binary image with given connectedness metric and number of iterations.
*/
int objectrecognition::dilation(cv::Mat& src, cv::Mat& dst, int connectedness, int iterations)
{
	src.copyTo(dst);
	while (iterations != 0) {
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (i == 0 || i == src.rows - 1 || j == 0 || j == src.cols - 1) {
					dst.at<uchar>(i, j) = 0;
					continue;
				}
				if (connectedness == 8) {
					if ((short)src.at<uchar>(i - 1, j - 1) == 255 || (short)src.at<uchar>(i - 1, j) == 255 || (short)src.at<uchar>(i - 1, j + 1) == 255 || (short)src.at<uchar>(i, j - 1) == 255 || (short)src.at<uchar>(i, j + 1)
						== 255 || (short)src.at<uchar>(i + 1, j - 1) == 255 || (short)src.at<uchar>(i + 1, j) == 255 || (short)src.at<uchar>(i + 1, j + 1) == 255) {
						dst.at<uchar>(i, j) = 255;
					}
				}
				else {
					if ((short)src.at<uchar>(i - 1, j) == 255 || (short)src.at<uchar>(i, j - 1) == 255 || (short)src.at<uchar>(i, j + 1)
						== 255 || (short)src.at<uchar>(i + 1, j) == 255) {
						dst.at<uchar>(i, j) = 255;
					}/*
					else {
						dst.at<uchar>(i, j) = 0;
					}*/
				}
			}
		}
		iterations--;
	}
	return 0;
}

/*
* Helper method that implements grassfire transform to fill holes in the image.
*/
int objectrecognition::grassfireTransform(cv::Mat& src, cv::Mat& dst)
{
	// Create temporary images
	cv::Mat temp2;
	temp2 = Mat::zeros(src.size(), CV_32F); // Initialize output image to all zeros

	// First Pass
	// Iterate from top left to bottom right
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (i == 0 || i == src.rows - 1 || j == 0 || j == src.cols - 1) {
				temp2.at<float>(i, j) = src.at<uchar>(i, j);
				continue;
			}
			if (src.at<uchar>(i, j) == 0) {
				// If the current pixel is zero, set the output pixel to 0.
				temp2.at<float>(i, j) = 0;
			}
			else {
				// If the current pixel is non-zero
				float minNeighbor = INFINITY;
				// Previous Neighbors
				for (int k = -1; k <= 0; k++) {
					for (int l = -1; l <= 1; l++) {
						if (k == 0 && l == 0) {
							break;
						}
						float neighbor = src.at<uchar>(i + k, j + l);
						minNeighbor = min(minNeighbor, neighbor);
					}
				}
				temp2.at<float>(i, j) = minNeighbor + 1;
			}
		}
	}

	// Second Pass
	// Iterate from bottom left to top right
	for (int i = src.rows - 1; i >= 0; i--) {
		for (int j = src.cols - 1; j >= 0; j--) {
			if (i == 0 || i == src.rows - 1 || j == 0 || j == src.cols - 1) {
				temp2.at<float>(i, j) = src.at<uchar>(i, j);
				continue;
			}
			// For each pixel
			float currentContent = temp2.at<float>(i, j);
			float minNeighbor = INFINITY;
			// Previous Neighbors
			for (int k = 1; k >= 0; k--) {
				for (int l = 1; l >= -1; l--) {
					if (k == 0 && l == 0) {
						break;
					}
					float neighbor = src.at<uchar>(i + k, j + l);
					minNeighbor = std::min(minNeighbor, neighbor);
				}
			}
			temp2.at<float>(i, j) = std::min(currentContent, minNeighbor + 1);
		}
	}

	// Third Pass to set pixel values
	dst = cv::Mat(temp2.size(), CV_8UC1);
	for (int i = 0; i < temp2.rows; i++) {
		for (int j = 0; j < temp2.cols; j++) {
			float currentContent = temp2.at<float>(i, j);
			if (currentContent > 0) {
				dst.at<uchar>(i, j) = 255;
			}
			else {
				dst.at<uchar>(i, j) = 0;
			}
		}
	}
	return 0;
}

/*
* Helper function to segment the image into regions.
*/
int objectrecognition::segmentImage(cv::Mat& src, cv::Mat& labels, cv::Mat& stats, cv::Mat& centroids, std::vector<Vec3b>& colors)
{
	int connectivity = 4;
	int n = 1;
	int numlabels = cv::connectedComponentsWithStats(src, labels, stats, centroids, 8, CV_32S);
	if (colors.size() < numlabels) {
		colors.clear();
		colors.push_back(Vec3b(0, 0, 0));
		for (int i = 1; i < numlabels; i++)
		{
			colors.push_back(Vec3b(0, 0, 0));
			if (stats.at<int>(i, CC_STAT_AREA) > minNumPixels)
			{
				colors.at(i) = Vec3b(rand() % 256, rand() % 256, rand() % 256);
			}
		}
	}
	return numlabels;
}

/*
* Helper function to highlight only the selected region.
*/
int objectrecognition::selectRegion(cv::Mat& src, cv::Mat& dst, int numlabels, int selected, cv::Mat& labels, cv::Mat& stats, std::vector<Vec3b>& colors) {
	Mat colored_img = Mat::zeros(src.size(), CV_8UC3);
	try {
		for (int i = 0; i < colored_img.rows; i++)
		{
			for (int j = 0; j < colored_img.cols; j++)
			{
				if (selected == 0) {
					int label = labels.at<int>(i, j);
					colored_img.at<Vec3b>(i, j) = colors[label];
				}
				else {
					int label = labels.at<int>(i, j);
					if (label == selected) {
						colored_img.at<Vec3b>(i, j) = colors[label];
					}
				}
			}
		}
	}
	catch (Exception e) {
		std::cerr << e.what() << std::endl;
	}
	colored_img.copyTo(dst);
	return 0;
}
