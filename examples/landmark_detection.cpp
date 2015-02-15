/*
 * superviseddescent: A C++11 implementation of the supervised descent
 *                    optimisation method
 * File: examples/landmark_detection.cpp
 *
 * Copyright 2014, 2015 Patrik Huber
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "superviseddescent/superviseddescent.hpp"
#include "superviseddescent/regressors.hpp"

extern "C" {
	#include "hog.h" // From the VLFeat C library
}

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/archive/text_oarchive.hpp"

#include <vector>
#include <iostream>
#include <fstream>
#include <utility>
#include <cassert>

using namespace superviseddescent;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using cv::Mat;
using cv::Vec2f;
using std::vector;
using std::string;
using std::cout;
using std::endl;

/**
 * Reads an ibug .pts landmark file and returns an ordered vector with
 * the 68 2D landmark coordinates.
 *
 * @param[in] filename Path to a .pts file.
 * @return An ordered vector with the 68 ibug landmarks.
 */
vector<Vec2f> readPtsLandmarks(std::string filename)
{
	using std::getline;
	vector<Vec2f> landmarks;
	landmarks.reserve(68);

	std::ifstream file(filename);
	if (!file.is_open()) {
		throw std::runtime_error(string("Could not open landmark file: " + filename));
	}
	
	string line;
	// Skip the first 3 lines, they're header lines:
	getline(file, line); // 'version: 1'
	getline(file, line); // 'n_points : 68'
	getline(file, line); // '{'

	while (getline(file, line))
	{
		if (line == "}") { // end of the file
			break;
		}
		std::stringstream lineStream(line);
		Vec2f landmark(0.0f, 0.0f);
		if (!(lineStream >> landmark[0] >> landmark[1])) {
			throw std::runtime_error(string("Landmark format error while parsing the line: " + line));
		}
		// From the iBug website:
		// "Please note that the re-annotated data for this challenge are saved in the Matlab convention of 1 being 
		// the first index, i.e. the coordinates of the top left pixel in an image are x=1, y=1."
		// ==> So we shift every point by 1:
		landmark[0] -= 1.0f;
		landmark[1] -= 1.0f;
		landmarks.emplace_back(landmark);
	}
	return landmarks;
};

// Loads images and their corresponding landmarks from .pts files into memory
/**
 * Loads all .png images and their corresponding .pts landmarks from the given
 * directory and returns them. 
 *
 * @param[in] directory A directory with .png images and ibug .pts files.
 * @return A pair with the loaded images and their landmarks (all in one cv::Mat).
 */
std::pair<vector<Mat>, Mat> loadIbugData(fs::path directory)
{
	vector<Mat> images;
	Mat landmarks;

	// Get all the filenames in the given directory:
	vector<fs::path> imageFilenames;
	fs::directory_iterator end_itr;
	for (fs::directory_iterator i(directory); i != end_itr; ++i)
	{
		if (fs::is_regular_file(i->status()) && i->path().extension() == ".png")
			imageFilenames.emplace_back(i->path());
	}

	// Load each image and its landmarks into memory:
	for (auto file : imageFilenames)
	{
		images.emplace_back(cv::imread(file.string()));
		// We load the landmarks and convert them into [x_0, ..., x_n, y_0, ..., y_n] format:
		file.replace_extension(".pts");
		auto lms = readPtsLandmarks(file.string());
		int numLandmarks = static_cast<int>(lms.size());
		Mat landmarksAsRow(1, 2 * numLandmarks, CV_32FC1);
		for (int i = 0; i < numLandmarks; ++i) {
			landmarksAsRow.at<float>(i) = lms[i][0]; // the x coordinate
			landmarksAsRow.at<float>(i + numLandmarks) = lms[i][1]; // y coordinate
		}
		landmarks.push_back(landmarksAsRow);
	}
	return std::make_pair(images, landmarks);
};

/**
 * Function object that extracts HoG features at given 2D landmark locations
 * and returns them as a row vector.
 *
 * We wrap all the C-style memory allocations of the VLFeat library
 * in cv::Mat's.
 * Note: Any other library and features can of course be used.
 */
class HogTransform
{
public:
	/**
	 * Constructs a HoG transform with given images and parameters.
	 * The images are needed so the HoG features can be extracted from the images
	 * when the SupervisedDescentOptimiser calls this functor (the optimiser
	 * itself doesn't know and care about the images).
	 *
	 * Note: \p images can consist of only 1 image, when using the model for
	 * prediction on new images.
	 *
	 * Note: Only VlHogVariantUoctti is tested.
	 *
	 * @param[in] images A vector of images used for training or testing.
	 * @param[in] vlHogVariant The VLFeat HoG variant.
	 * @param[in] numCells Number of HoG cells that should be constructed around each landmark.
	 * @param[in] cellSize Width of one HoG cell in pixels.
	 * @param[in] numBins Number of orientations of a HoG cell.
	 */
	HogTransform(vector<Mat> images, VlHogVariant vlHogVariant, int numCells, int cellSize, int numBins) : images(images), vlHogVariant(vlHogVariant), numCells(numCells), cellSize(cellSize), numBins(numBins)
	{
	};

	/**
	 * Uses the current parameters (the 2D landmark locations, in SDM
	 * terminology the \c x), and extracts HoG features at these positions.
	 * These HoG features are the new \c y values.
	 *
	 * The 2D landmark position estimates are given as a row vector
	 * [x_0, ..., x_n, y_0, ..., y_n].
	 *
	 * @param[in] parameters The current 2D landmark position estimate.
	 * @param[in] regressorLevel Not used in this example.
	 * @param[in] trainingIndex Gets supplied by the SupervisedDescent optimiser during training and testing, to know from which image to extract features.
	 * @return Returns the HoG features at the given 2D locations.
	 */
	cv::Mat operator()(cv::Mat parameters, size_t regressorLevel, int trainingIndex = 0)
	{
		assert(parameters.rows == 1);
		using cv::Mat;
			
		Mat grayImage;
		if (images[trainingIndex].channels() == 3) {
			cv::cvtColor(images[trainingIndex], grayImage, cv::COLOR_BGR2GRAY);
		}
		else {
			grayImage = images[trainingIndex];
		}

		// Note: We could use the 'regressorLevel' to choose the window size (and
		// other parameters adaptively). We omit this for the sake of a short example.

		int patchWidthHalf = numCells * (cellSize / 2);

		Mat hogDescriptors; // We'll get the dimensions later from vl_hog_get_*

		int numLandmarks = parameters.cols / 2;
		for (int i = 0; i < numLandmarks; ++i) {
			int x = cvRound(parameters.at<float>(i));
			int y = cvRound(parameters.at<float>(i + numLandmarks));

			Mat roiImg;
			if (x - patchWidthHalf < 0 || y - patchWidthHalf < 0 || x + patchWidthHalf >= grayImage.cols || y + patchWidthHalf >= grayImage.rows) {
				// The feature extraction location is too far near a border. We extend the
				// image (add a black canvas) and then extract from this larger image.
				int borderLeft = (x - patchWidthHalf) < 0 ? std::abs(x - patchWidthHalf) : 0; // x and y are patch-centers
				int borderTop = (y - patchWidthHalf) < 0 ? std::abs(y - patchWidthHalf) : 0;
				int borderRight = (x + patchWidthHalf) >= grayImage.cols ? std::abs(grayImage.cols - (x + patchWidthHalf)) : 0;
				int borderBottom = (y + patchWidthHalf) >= grayImage.rows ? std::abs(grayImage.rows - (y + patchWidthHalf)) : 0;
				Mat extendedImage = grayImage.clone();
				cv::copyMakeBorder(extendedImage, extendedImage, borderTop, borderBottom, borderLeft, borderRight, cv::BORDER_CONSTANT, cv::Scalar(0));
				cv::Rect roi((x - patchWidthHalf) + borderLeft, (y - patchWidthHalf) + borderTop, patchWidthHalf * 2, patchWidthHalf * 2); // Rect: x y w h. x and y are top-left corner.
				roiImg = extendedImage(roi).clone(); // clone because we need a continuous memory block
			}
			else {
				cv::Rect roi(x - patchWidthHalf, y - patchWidthHalf, patchWidthHalf * 2, patchWidthHalf * 2); // x y w h. Rect: x and y are top-left corner. Our x and y are center. Convert.
				roiImg = grayImage(roi).clone(); // clone because we need a continuous memory block
			}
			roiImg.convertTo(roiImg, CV_32FC1); // vl_hog_put_image expects a float* (values 0.0f-255.0f)
			VlHog* hog = vl_hog_new(vlHogVariant, numBins, false); // transposed (=col-major) = false
			vl_hog_put_image(hog, (float*)roiImg.data, roiImg.cols, roiImg.rows, 1, cellSize); // (the '1' is numChannels)
			int ww = static_cast<int>(vl_hog_get_width(hog)); // assert ww == hh == numCells
			int hh = static_cast<int>(vl_hog_get_height(hog));
			int dd = static_cast<int>(vl_hog_get_dimension(hog)); // assert ww=hogDim1, hh=hogDim2, dd=hogDim3
			Mat hogArray(1, ww*hh*dd, CV_32FC1); // safer & same result. Don't use C-style memory management.
			vl_hog_extract(hog, hogArray.ptr<float>(0));
			vl_hog_delete(hog);
			Mat hogDescriptor(hh*ww*dd, 1, CV_32FC1);
			// Stack the third dimensions of the HOG descriptor of this patch one after each other in a column-vector:
			for (int j = 0; j < dd; ++j) {
				Mat hogFeatures(hh, ww, CV_32FC1, hogArray.ptr<float>(0) + j*ww*hh); // Creates the same array as in Matlab. I might have to check this again if hh!=ww (non-square)
				hogFeatures = hogFeatures.t(); // necessary because the Matlab reshape() takes column-wise from the matrix while the OpenCV reshape() takes row-wise.
				hogFeatures = hogFeatures.reshape(0, hh*ww); // make it to a column-vector
				Mat currentDimSubMat = hogDescriptor.rowRange(j*ww*hh, j*ww*hh + ww*hh);
				hogFeatures.copyTo(currentDimSubMat);
			}
			hogDescriptor = hogDescriptor.t(); // now a row-vector
			hogDescriptors.push_back(hogDescriptor);
		}
		// concatenate all the descriptors for this sample vertically (into a row-vector):
		hogDescriptors = hogDescriptors.reshape(0, hogDescriptors.cols * numLandmarks).t();
		return hogDescriptors;
	};

private:
	vector<Mat> images;
	VlHogVariant vlHogVariant;
	int numCells;
	int cellSize;
	int numBins;
};

/**
 * Load the pre-calculated landmarks mean from the filesystem.
 *
 * @param[in] filename Path to a file with mean landmarks.
 * @return A cv::Mat of the loaded mean model.
 */
cv::Mat loadMean(fs::path filename)
{
	std::ifstream file(filename.string());
	if (!file.is_open()) {
		throw std::runtime_error(string("Could not open file: " + filename.string()));
	}

	string line;
	getline(file, line);

	vector<string> values;
	boost::split(values, line, boost::is_any_of(","));

	int twiceNumLandmarks = static_cast<int>(values.size());
	Mat mean(1, twiceNumLandmarks, CV_32FC1);
	for (int i = 0; i < twiceNumLandmarks; ++i) {
		mean.at<float>(i) = std::stof(values[i]);
	}

	return mean;
};

/**
 * Performs an initial alignment of the model, by putting the mean model into
 * the center of the face box.
 *
 * An optional scaling and translation parameters can be given to generate
 * perturbations of the initialisation.
 *
 * @param[in] mean Mean model points.
 * @param[in] faceBox A facebox to align the model to.
 * @param[in] scalingX Optional scaling in x of the model.
 * @param[in] scalingY Optional scaling in y of the model.
 * @param[in] translationX Optional translation in x of the model.
 * @param[in] translationY Optional translation in y of the model.
 * @return A cv::Mat of the aligned points.
 */
Mat alignMean(Mat mean, cv::Rect faceBox, float scalingX=1.0f, float scalingY=1.0f, float translationX=0.0f, float translationY=0.0f)
{
	// Initial estimate x_0: Center the mean face at the [-0.5, 0.5] x [-0.5, 0.5] square (assuming the face-box is that square)
	// More precise: Take the mean as it is (assume it is in a space [-0.5, 0.5] x [-0.5, 0.5]), and just place it in the face-box as
	// if the box is [-0.5, 0.5] x [-0.5, 0.5]. (i.e. the mean coordinates get upscaled)
	Mat alignedMean = mean.clone();
	Mat alignedMeanX = alignedMean.colRange(0, alignedMean.cols / 2);
	Mat alignedMeanY = alignedMean.colRange(alignedMean.cols / 2, alignedMean.cols);
	alignedMeanX = (alignedMeanX*scalingX + 0.5f + translationX) * faceBox.width + faceBox.x;
	alignedMeanY = (alignedMeanY*scalingY + 0.5f + translationY) * faceBox.height + faceBox.y;
	return alignedMean;
}

/**
 * Draws the given landmarks into the image.
 *
 * @param[in] image An image to draw into.
 * @param[in] landmarks The landmarks to draw.
 * @param[in] color Color of the landmarks to be drawn.
 */
void drawLandmarks(cv::Mat image, cv::Mat landmarks, cv::Scalar color = cv::Scalar(0.0, 255.0, 0.0))
{
	auto numLandmarks = std::max(landmarks.cols, landmarks.rows) / 2;
	for (int i = 0; i < numLandmarks; ++i) {
		cv::circle(image, cv::Point2f(landmarks.at<float>(i), landmarks.at<float>(i + numLandmarks)), 2, color);
	}
}

/**
 * This app demonstrates learning of the descent direction from data for
 * a simple facial landmark detection sample app.
 *
 * First, we detect a face using OpenCV's face detector, and put the mean
 * landmarks into the face box. Then, the update step of the landmark coordinates
 * is learned using cascaded regression. HoG features are extracted around the
 * landmark positions, and from that, the update step towards the ground truth
 * positions is learned.
 *
 * This is an example of the library when a known template \c y is not available
 * during testing (because the HoG features are different for every subject).
 * Learning is thus performed without a \c y.
 */
int main(int argc, char *argv[])
{
	fs::path trainingset, meanfile, facedetector;
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"display the help message")
			("data,d", po::value<fs::path>(&trainingset)->required()->default_value("data/ibug_lfpw_trainset"),
				"path to ibug LFPW example images and landmarks")
			("mean,m", po::value<fs::path>(&meanfile)->required()->default_value("data/mean_ibug_lfpw_68.txt"),
				"pre-calculated mean from ibug LFPW")
			("facedetector,f", po::value<fs::path>(&facedetector)->required(),
				"full path to OpenCV's face detector (haarcascade_frontalface_alt2.xml)")
			;
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: landmark_detection [options]" << endl;
			cout << desc;
			return EXIT_SUCCESS;
		}
		po::notify(vm);
	}
	catch (po::error& e) {
		cout << "Error while parsing command-line arguments: " << e.what() << endl;
		cout << "Use --help to display a list of options." << endl;
		return EXIT_SUCCESS;
	}

	vector<Mat> trainingImages;
	Mat trainingLandmarks;
	try	{
		std::tie(trainingImages, trainingLandmarks) = loadIbugData(trainingset);
	}
	catch (const fs::filesystem_error& ex)
	{
		cout << ex.what() << endl;
		return EXIT_FAILURE;
	}

	// Load the pre-calculated (and scaled) mean of all landmarks:
	Mat modelMean = loadMean(meanfile);
	
	// Load the face detector from OpenCV:
	cv::CascadeClassifier faceCascade;
	if (!faceCascade.load(facedetector.string()))
	{
		cout << "Error loading face detection model." << endl;
		return EXIT_FAILURE;
	}
	
	// Run the face detector and obtain the initial estimate x0 using the mean landmarks:
	Mat x0;
	for (size_t i = 0; i < trainingImages.size(); ++i) {
		vector<cv::Rect> detectedFaces;
		faceCascade.detectMultiScale(trainingImages[i], detectedFaces, 1.2, 2, 0, cv::Size(50, 50));
		// In a real application, you should verify that the detected face is not a false positive
		x0.push_back(alignMean(modelMean, cv::Rect(detectedFaces[0])));
	}

	// We might want to augment the training set by perturbing the
	// initialisations, which we skip here.

	// Create 3 regularised linear regressors in series:
	vector<LinearRegressor> regressors;
	regressors.emplace_back(LinearRegressor(Regulariser(Regulariser::RegularisationType::MatrixNorm, 2.0f, true)));
	regressors.emplace_back(LinearRegressor(Regulariser(Regulariser::RegularisationType::MatrixNorm, 2.0f, true)));
	regressors.emplace_back(LinearRegressor(Regulariser(Regulariser::RegularisationType::MatrixNorm, 2.0f, true)));
	
	SupervisedDescentOptimiser<LinearRegressor> supervisedDescentModel(regressors);
	
	HogTransform hog(trainingImages, VlHogVariant::VlHogVariantUoctti, 3 /*numCells*/, 12 /*cellSize*/, 4 /*numBins*/);

	// Train the model. We'll also specify an optional callback function:
	cout << "Training the model, printing the residual after each learned regressor: " << endl;
	auto printResidual = [&trainingLandmarks](const cv::Mat& currentPredictions) {
		cout << cv::norm(currentPredictions, trainingLandmarks, cv::NORM_L2) / cv::norm(trainingLandmarks, cv::NORM_L2) << endl;
	};
	
	supervisedDescentModel.train(trainingLandmarks, x0, Mat(), hog, printResidual);

	// To test on a whole bunch of images, we could do something roughly like this:
	// supervisedDescentModel.test(x0_ts,
	//	Mat(),
	//	HogTransform(testImages, VlHogVariant::VlHogVariantUoctti, 3, 12, 4),
	//	[&groundtruth_ts](const cv::Mat& currentPredictions) { std::cout << cv::norm(currentPredictions, groundtruth_ts, cv::NORM_L2) / cv::norm(groundtruth_ts, cv::NORM_L2) << std::endl; }
	// );
	
	// Detect the landmarks on a single image:
	Mat image = cv::imread((trainingset / fs::path("image_0005.png")).string());
	vector<cv::Rect> detectedFaces;
	faceCascade.detectMultiScale(image, detectedFaces, 1.2, 2, 0, cv::Size(50, 50));
	Mat initialAlignment = alignMean(modelMean, cv::Rect(detectedFaces[0]));
	Mat prediction = supervisedDescentModel.predict(initialAlignment, Mat(), HogTransform({ image }, VlHogVariant::VlHogVariantUoctti, 3, 12, 4));
	drawLandmarks(image, prediction, { 0, 255, 255 });
	cv::imwrite("out.png", image);

	// Save the learned model:
	std::ofstream learnedModelFile("landmark_regressor_ibug_68lms.txt");
	boost::archive::text_oarchive oa(learnedModelFile);
	oa << supervisedDescentModel;

	return EXIT_SUCCESS;
}
