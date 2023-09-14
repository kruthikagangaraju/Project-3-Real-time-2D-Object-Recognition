/*
  Malhar Mahant & Kruthika Gangaraju
  SP23

  Implements functions to generate different kinds of features from an image and find similar images using generated features.
*/
#include <cstring>
#include <cmath>
#include <dirent.h>
#include "csv_util.h"
#include "filters.h"
#include "matchfunctions.h"

#pragma warning(disable : 4996)

// Feature files
char standardDeviationsFile[256] = "standardDeviations.csv";
char labelToFeaturesFile[256] = "labelToFeatures.csv";
char labelMeanAndStdDev[256] = "labelMeanAndStdDev.csv";
// Label name for standardDeviations.csv file
char name[256] = "stdDevMeans";

// Minimum area of pixels to consider a valid object
int minNumPixel = 2048;

/*
* Helper method to check if a file exists.
*/
bool checkIfFileExists(char* fpath) {
	FILE* fp;
	fp = fopen(fpath, "r");
	if (!fp) {
		return false;
	}
	if (fp) {
		fclose(fp);
		return true;
	}
}

/*
* Helper method to calculate the Scaled Euclidean Distance between 2 given vectors of features.
*/
float calculateScaledEuclideanDistance(std::vector<float>& featuresA, std::vector<float>& featuresB) {
	std::vector<char*> labels;
	std::vector<std::vector<float>> stdDevMeanData;
	read_image_data_csv(standardDeviationsFile, labels, stdDevMeanData, 0);
	std::vector<float> stdDevMeanDataVector = stdDevMeanData.at(0);
	float scaledEuclideanDistance = 0;
	for (int i = 0; i < featuresA.size(); i++) {
		float diff = featuresA.at(i) - featuresB.at(i);
		// To treat mirror images as the same object
		if (i == 0 || i == 1 || i == featuresA.size() - 1) {
			diff = abs(featuresA.at(i)) - abs(featuresB.at(i));
		}
		float error = diff / stdDevMeanDataVector.at(i * 2);
		scaledEuclideanDistance = scaledEuclideanDistance + (error * error);
	}
	return scaledEuclideanDistance;
}

/*
* Helper method to find contours of objects in the binary image
*/
int matchfunctions::findAllContours(cv::Mat& src, std::vector<std::vector<Point>>& contours, std::vector<Vec4i>& hierarchy) {
    // Find contours in the binary image
    cv::findContours(src, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	return 0;
}

/*
* Helper method to generate features to be used for classification of objects
*/
int matchfunctions::generateFeatures(std::vector<Point>& contour, std::vector<float>& features) {
	cv::Moments moments = cv::moments(contour);
	// Calculate Hu Moments 
	double huMoments[7];
	/*
	* The third moment: This measures the skewness of the image. The third moment provides information about the orientation of the object in the image.
	* Positive-Negative signs describe mirror images. So we take absolute values to consider mirror images as the same.
	*/
	features.push_back(abs(moments.nu30));
	features.push_back(abs(moments.nu03));
	cv::HuMoments(moments, huMoments);
	// Log scale hu moments 
	for (int i = 0; i < 7; i++) {
		huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]));
		if (i == 6)
			features.push_back(abs(huMoments[i]));
		else
			features.push_back(huMoments[i]);
	}
	return 0;
}

/*
* Helper method to calculate the mean value of given float values in a vector.
*/
double computeMean(std::vector<float>& distances) {
	float sum = 0.0;
	for (float d : distances) {
		sum += d;
	}
	return sum / distances.size();
}

/*
* Helper method to calculate the standard deviationof given float values in a vector and a given mean.
*/
double computeStd(std::vector<float>& data, float mean) {
	double variance = 0.0;
	for (double d : data) {
		variance += pow(d - mean, 2.0);
	}
	variance /= data.size();
	return sqrt(variance);
}

/*
* Compute the mean and standard deviation of distances in a distribution of given data points.
*/
std::vector<float> computeMeanStdDistances(std::vector<std::vector<float>>& datapoints) {
	std::vector<float> distances;
	for (size_t i = 0; i < datapoints.size(); i++) {
		for (size_t j = 0; j < datapoints.size(); j++) {
			if (i != j) {
				float distance = calculateScaledEuclideanDistance(datapoints[i], datapoints[j]);
				distances.push_back(distance);
			}
		}
	}

	float meanDistance = computeMean(distances);
	float stdDevDistance = computeStd(distances, meanDistance);
	std::vector<float> result;
	result.push_back(meanDistance);
	result.push_back(stdDevDistance);
	return result;
}

/*
* Helper function to calculate the mean and standard deviation of distances in the distribution of data points for each label.
*/
void calculateAndSaveLabelMeanAndStdDev() {
	bool fileExists = checkIfFileExists(labelMeanAndStdDev);
	// Read Label to Features file
	std::vector<char*> labels;
	std::vector<std::vector<float>> labelToFeatures;
	read_image_data_csv(labelToFeaturesFile, labels, labelToFeatures, 0);

	// Accumulate all training instances for each label category
	std::map<std::string, std::vector<std::vector<float>>> map;
	for (int j = 0; j < labels.size(); j++) {
		std::vector<float> labelFeatures = labelToFeatures.at(j);
		char* label = labels.at(j);
		std::vector<std::vector<float>> allTrainingImagesFeatures;
		/*if (map.find(label) != map.end()) {
			std::cout << "In ghere: " << std::endl;*/
			allTrainingImagesFeatures = map[label];
		//}
		allTrainingImagesFeatures.push_back(labelFeatures);
		map[label] = /*map.emplace(label, */allTrainingImagesFeatures;
	}
	// Calculate Mean and Std Dev distance between items in each class
	int flag = 1;
	for (auto it = map.begin(); it != map.end(); ++it) {
		std::string label = it->first;
		char* itLabel = new char[256];
		strcpy(itLabel, label.c_str());
		std::vector<std::vector<float>> allTrainingImagesFeatures = it->second;
		std::vector<float> meanAndStdDevPair = computeMeanStdDistances(allTrainingImagesFeatures);
		meanAndStdDevPair.push_back(allTrainingImagesFeatures.size());
		append_image_data_csv(labelMeanAndStdDev, itLabel, meanAndStdDevPair, flag);
		if (flag) {
			flag = 0;
		}
	}
	return;
}

/*
* Helper function to calculate the mean and standard deviation of features across all labels.
*/
void calculateNewStdDevAndMean(std::vector<std::vector<float>> labelToFeatures) {
	if (!labelToFeatures.empty()) {
		std::vector<float> stdDevMeanFeatures;
		stdDevMeanFeatures.resize((labelToFeatures[0].size() * 2) + 1, 0);
		float n = 0;
		std::vector<float> sums(labelToFeatures[0].size(), 0);
		for (auto dbIter = labelToFeatures.begin(); dbIter != labelToFeatures.end(); ++dbIter) {
			std::vector<float> features = *dbIter;
			int i = 0;
			for (auto featureItr = features.begin(); featureItr != features.end(); ++featureItr) {
				// Calculate new mean
				float oldMean = stdDevMeanFeatures.at((i * 2) + 1);
				float newMean = (n * oldMean + features.at(i)) / (n + 1);
				// Calculate new std dev
				float oldStd = stdDevMeanFeatures.at(i * 2);
				float variation = (*featureItr - newMean);
				float newStd = sqrtf(((n * oldStd * oldStd) + variation * variation) / (n + 1));
				stdDevMeanFeatures.at(i * 2) = newStd;
				stdDevMeanFeatures.at((i * 2) + 1) = newMean;
				i++;
			}
			n++;
		}
		stdDevMeanFeatures.at(stdDevMeanFeatures.size() - 1) = n;
		append_image_data_csv(standardDeviationsFile, name, stdDevMeanFeatures, 1);
	}
}

/*
* Generates and saves the features for a given contour for a given label in the training data.
*/
int matchfunctions::generateAndSaveFeatures(std::string label, std::vector<Point>& contour) {
	std::vector<float> features;
	matchfunctions::generateFeatures(contour, features);
	char* labelChar = new char[256];
	strcpy(labelChar, label.c_str());
	if (!checkIfFileExists(standardDeviationsFile)) {
		std::vector<float> stdDevMeanFeatures;
		stdDevMeanFeatures.resize((features.size() * 2) + 1, 0);
		append_image_data_csv(standardDeviationsFile, labelChar, stdDevMeanFeatures, 1);
	}
	std::vector<char*> labels;
	std::vector<std::vector<float>> stdDevMeanData;
	read_image_data_csv(standardDeviationsFile, labels, stdDevMeanData, 0);
	std::vector<float> stdDevMeanDataVector = stdDevMeanData.at(0);
	float n = stdDevMeanDataVector.at(stdDevMeanDataVector.size() - 1);
	labels.clear();
	std::vector<std::vector<float>> labelToFeatures;
	read_image_data_csv(labelToFeaturesFile, labels, labelToFeatures, 0);
	if (n != labelToFeatures.size()) {
		calculateNewStdDevAndMean(labelToFeatures);
		labels.clear();
		stdDevMeanData.clear();
		read_image_data_csv(standardDeviationsFile, labels, stdDevMeanData, 0);
		stdDevMeanDataVector = stdDevMeanData.at(0);
		n = stdDevMeanDataVector.at(stdDevMeanDataVector.size() - 1);
	}
	// For each feature
	for (int i = 0; i < features.size(); i++) {
		// Calculate new mean
		float oldMean = stdDevMeanDataVector.at((i * 2)+ 1);
		float newMean = (n * oldMean + features.at(i)) / (n + 1);
		// Calculate new std dev
		float oldStd = stdDevMeanDataVector.at(i * 2);
		float variation = (features.at(i) - newMean);
		float newStd = sqrtf(((n * oldStd * oldStd) + variation * variation) / (n + 1));
		stdDevMeanDataVector.at(i * 2) = newStd;
		stdDevMeanDataVector.at((i * 2) + 1) = newMean;
	}
	stdDevMeanDataVector.at(stdDevMeanDataVector.size() - 1) = n + 1;
	append_image_data_csv(standardDeviationsFile, name, stdDevMeanDataVector, 1);
	append_image_data_csv(labelToFeaturesFile, labelChar, features, 0);
	calculateAndSaveLabelMeanAndStdDev();
}

/*
* Classify new objects in image using Nearest Neighbor Classification.
*/
int matchfunctions::nearestNeighbor(std::vector<std::vector<Point>>& contours, std::vector<Vec4i>& hierarchy, std::vector<std::string>& outLabels) {
	std::vector<char*> labels;
	std::vector<std::vector<float>> featuresData;
	outLabels.resize(contours.size(), "");
	// Read features from CSV
	if (read_image_data_csv(labelToFeaturesFile, labels, featuresData, 0) >= 0) {
		for (size_t i = 0; i < contours.size(); i++)
		{
			if (cv::contourArea(contours[i]) > minNumPixel) {
				std::vector<float> features;
				matchfunctions::generateFeatures(contours[i], features);
				// Map of filenames to errors
				std::multimap<float, char*> map;
				for (int j = 0; j < labels.size(); j++) {
					// Calculate distance for each image
					std::vector<float> labelFeatures = featuresData.at(j);
					float error = calculateScaledEuclideanDistance(features, labelFeatures);
					map.insert({ error, labels.at(j) });
				}
				outLabels.at(i) = map.begin()->second;
			}
		}	
		return(0);
	}
	return -1;
}

/*
* Classify new objects in image using Multi-Class K Nearest Neighbor Classification.
*/
int matchfunctions::kNearestNeighbor(std::vector<std::vector<Point>>& contours, std::vector<Vec4i>& hierarchy, std::vector<std::string>& outLabels, std::vector <std::string>& sumDistances) {
	std::vector<char*> labels;
	std::vector<std::vector<float>> featuresData;
	outLabels.resize(contours.size(), "");
	sumDistances.resize(contours.size(), "");
	// Read features from CSV
	if (read_image_data_csv(labelToFeaturesFile, labels, featuresData, 0) >= 0) {
		for (size_t i = 0; i < contours.size(); i++)
		{
			if (cv::contourArea(contours[i]) > minNumPixel) {
				std::vector<float> features;
				matchfunctions::generateFeatures(contours[i], features);
				// Map of filenames to errors
				std::map<std::string, std::vector<float>> map;
				for (int j = 0; j < labels.size(); j++) {
					// Calculate distance for each image
					std::vector<float> labelFeatures = featuresData.at(j);
					float error = calculateScaledEuclideanDistance(features, labelFeatures);
					std::string label = labels.at(j);
					std::vector<float> distances;
					if (map.find(label) != map.end()) {
						distances = map.at(label);
					}
					distances.push_back(error);
					map.insert({ labels.at(j), distances });
				}

				// For each label find the sum of distance to k-nearest neighbors
				std::multimap<float, std::string> distanceToClassMap;
				std::map<char*, std::vector<float>>::iterator itr;
				for (auto itr = map.begin(); itr != map.end(); itr++) {
					std::string label = itr->first;
					std::vector<float> distances = itr->second;
					sort(distances.begin(), distances.end());
					float distanceToClass = 0;
					int j = 0;
					float lastDistance = 0;
					for (auto it = distances.begin(); j < 5/* && it != distances.end()*/; j++) {
						// Handles where K samples for a class are not present in training data
						if (j < distances.size()) {
							lastDistance = *it;
							distanceToClass += lastDistance;
							it++;
						}
						else {
							distanceToClass += lastDistance;
						}
					}
					distanceToClassMap.insert({ distanceToClass, label });
				}

				for (auto it = distanceToClassMap.begin(); it != distanceToClassMap.end(); ++it) {
					std::cout << "Label: " << it->second << " Sum: " << it->first << std::endl;
				}

				// multimap are sorted by minimum key 
				auto it = distanceToClassMap.begin();
				sumDistances.at(i) = std::to_string(it->first);
				std::string predictedLabel = it->second;

				// Get Mean and Std Deviation of distribution within the predicted class
				std::vector<char*> mapLabels;
				std::vector <std::vector<float>> meanAndStdDevPairData;
				read_image_data_csv(labelMeanAndStdDev, mapLabels, meanAndStdDevPairData, 0);
				float meanOfClass = 0, stdDevOfClass = 0;
				for (int i = 0; i < mapLabels.size(); i++) {
					std::string label = mapLabels.at(i);
					if (predictedLabel == label) {
						std::vector<float> meanAndStdDevPair = meanAndStdDevPairData.at(i);
						meanOfClass = meanAndStdDevPair[0];
						stdDevOfClass = meanAndStdDevPair[1];
						break;
					}
				}

				// Get distance of this data from Kth nearest neighbor
				float distance;
				std::vector<float> distances = map[predictedLabel];
				if (distances.size() < 5) {
					distance = distances[distances.size() - 1];
				}
				else {
					distance = distances[5];
				}
				// Use the standard deviation and mean of distances in distribution of the predicted label as a threshold to gain confidence in prediction.
				if (abs(meanOfClass - distance) > 1.5 * stdDevOfClass) {
					outLabels.at(i) = "Unknown";
				}
				else {
					outLabels.at(i) = it->second;
				}
			}
		}
		return(0);
	}
	return -1;
}