/*
 * superviseddescent: A C++11 implementation of the supervised descent
 *                    optimisation method
 * File: apps/rcr/rcr-train-tracking.cpp
 *
 * Copyright 2015 Patrik Huber
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

#include "rcr/landmark.hpp"
#include "rcr/landmarks_io.hpp"
#include "rcr/adaptive_vlhog.hpp"
#include "rcr/helpers.hpp"
#include "rcr/model.hpp"

#include "cereal/cereal.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/info_parser.hpp"

#include <vector>
#include <iostream>
#include <fstream>
#include <utility>
#include <random>
#include <cassert>

using namespace superviseddescent;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace pt = boost::property_tree;
using cv::Mat;
using cv::Vec2f;
using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::size_t;

/**
 * Loads all .png images and their corresponding .pts landmarks from the given
 * directory and returns them. 
 *
 * @param[in] directory A directory with .png images and ibug .pts files.
 * @return A pair with the loaded images and their landmarks (all in one cv::Mat TODO update).
 */
std::pair<vector<Mat>, vector<rcr::LandmarkCollection<Vec2f>>> loadIbugData(fs::path directory)
{
	vector<Mat> images;
	vector<rcr::LandmarkCollection<Vec2f>> landmarks;

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
		auto lms = rcr::read_pts_landmarks(file.string());
		landmarks.emplace_back(lms);
	}
	return std::make_pair(images, landmarks);
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
 * Perturb...
 *
 * tx, ty are in percent of the total face box width/height.
 *
 * @param[in] facebox A facebox to align the model to.
 * @param[in] translationX Translation in x of the box.
 * @param[in] translationY Translation in y of the box.
 * @param[in] scaling Optional scale factor of the box.
 * @return A perturbed cv::Rect.
 */
cv::Rect perturb(cv::Rect facebox, float translationX, float translationY, float scaling = 1.0f)
{
	
	auto tx_pixel = translationX * facebox.width;
	auto ty_pixel = translationY * facebox.height;
	// Because the reference point is on the top left and not in the center, we
	// need to temporarily store the new width and calculate half of the offset.
	// We need to move it further to compensate for the scaling, i.e. keep the center the center.
	auto perturbed_width = facebox.width * scaling;
	auto perturbed_height = facebox.height * scaling;
	//auto perturbed_width_diff = facebox.width - perturbed_width;
	//auto perturbed_height_diff = facebox.height - perturbed_height;
	// Note: Rounding?
	cv::Rect perturbedBox(facebox.x + (facebox.width - perturbed_width) / 2.0f + tx_pixel, facebox.y + (facebox.height - perturbed_height) / 2.0f + ty_pixel, perturbed_width, perturbed_height);

	return perturbedBox;
}

// Error evaluation (average pixel distance (L2 norm) of all LMs, normalised by IED):
// could make C++14 generic lambda, or templated function. Requires: cv::norm can handle the given T.
// Todo Rename to "lm1, lm2" or similar.
double norm(const rcr::Landmark<Vec2f>& prediction, const rcr::Landmark<Vec2f>& groundtruth)
{
	return cv::norm(prediction.coordinates, groundtruth.coordinates, cv::NORM_L2);
};

Mat elementwiseNorm(const rcr::LandmarkCollection<Vec2f>& prediction, const rcr::LandmarkCollection<Vec2f>& groundtruth)
{
	assert(prediction.size() == groundtruth.size());
	Mat result(1, prediction.size(), CV_32FC1); // a row with each entry a norm...
	for (std::size_t i = 0; i < prediction.size(); ++i) {
		result.at<float>(i) = norm(prediction[i], groundtruth[i]);
	}
	return result;
};

// After:
//double meanError = cv::mean(normalisedErrors)[0]; // = the mean over all, identical to simple case
//cv::reduce(normalisedErrors, normalisedErrors, 0, CV_REDUCE_AVG); // reduce to one row
Mat calculateNormalisedLandmarkErrors(Mat currentPredictions, Mat x_gt, vector<string> modelLandmarks, vector<string> rightEyeIdentifiers, vector<string> leftEyeIdentifiers)
{
	Mat normalisedErrors;
	for (int row = 0; row < currentPredictions.rows; ++row) {
		auto predc = rcr::to_landmark_collection(currentPredictions.row(row), modelLandmarks);
		auto grc = rcr::to_landmark_collection(x_gt.row(row), modelLandmarks);
		Mat err = elementwiseNorm(predc, grc).mul(1.0f / rcr::get_ied(predc, rightEyeIdentifiers, leftEyeIdentifiers));
		normalisedErrors.push_back(err);
	}
	return normalisedErrors;
};

vector<string> readLandmarksListToTrain(fs::path configfile)
{
	pt::ptree configTree;
	pt::read_info(configfile.string(), configTree); // Can throw a pt::ptree_error, maybe try/catch
	
	vector<string> modelLandmarks;
	// Get stuff from the modelLandmarks subtree:
	pt::ptree ptModelLandmarks = configTree.get_child("modelLandmarks");
	string modelLandmarksUsage = ptModelLandmarks.get<string>("landmarks");
	if (modelLandmarksUsage.empty()) {
		// value is empty, meaning it's a node and the user should specify a list of 'landmarks'
		pt::ptree ptModelLandmarksList = ptModelLandmarks.get_child("landmarks");
		for (const auto& kv : ptModelLandmarksList) {
			modelLandmarks.push_back(kv.first);
		}
		cout << "Loaded a list of " << modelLandmarks.size() << " landmarks to train the model." << endl;
	}
	else if (modelLandmarksUsage == "all") {
		throw std::logic_error("Using 'all' modelLandmarks is not implemented yet - specify a list for now.");
	}
	else {
		throw std::logic_error("Error reading the models 'landmarks' key, should either provide a node with a list of landmarks or specify 'all'.");
	}
	return modelLandmarks;
}

// Returns a pair of (vector<string> rightEyeIdentifiers, vector<string> leftEyeIdentifiers)
// throws ptree, logic? error
std::pair<vector<string>, vector<string>> readHowToCalculateTheIED(fs::path evaluationfile)
{
	vector<string> rightEyeIdentifiers, leftEyeIdentifiers;

	pt::ptree evalConfigTree;
	string rightEye;
	string leftEye;
	pt::read_info(evaluationfile.string(), evalConfigTree); // could throw a boost::property_tree::ptree_error, maybe try/catch

	pt::ptree ptParameters = evalConfigTree.get_child("interEyeDistance");
	rightEye = ptParameters.get<string>("rightEye");
	leftEye = ptParameters.get<string>("leftEye");
	
	// Process the interEyeDistance landmarks - one or two identifiers might be given
	boost::split(rightEyeIdentifiers, rightEye, boost::is_any_of(" "));
	boost::split(leftEyeIdentifiers, leftEye, boost::is_any_of(" "));
	return std::make_pair(rightEyeIdentifiers, leftEyeIdentifiers);
}

/**
 * This app implements the training of a robust cascaded regression landmark
 * detection model, similar to the one proposed in the paper:
 * "Random Cascaded-Regression Copse for Robust Facial Landmark Detection",
 * Z. Feng, P. Huber, J. Kittler, W. Christmas, X.J. Wu,
 * IEEE Signal Processing Letters, Vol:22(1), 2015.
 * 
 * However, the training is modified for tracking rather than detection.
 * 
 * The RCR landmark detection is one of the prime examples and motivation for
 * the supervised descent library.
 *
 * First, we detect a face using OpenCV's face detector, and put the mean
 * landmarks into the face box. Then, the update step of the landmark coordinates
 * is learned using cascaded regression. HoG features are extracted around the
 * landmark positions, and from that, the update step towards the ground truth
 * positions is learned.
 * The training data, or more precisely the training face boxes, are perturbed,
 * to create a more robust model.
 *
 * This is an example of the library when a known template \c y is not available
 * during testing (because the HoG features are different for every subject).
 * Learning is thus performed without a \c y.
 */
int main(int argc, char *argv[])
{
	fs::path trainingset, meanfile, configfile, evaluationfile, outputfile, testset;
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"display the help message")
			("data,d", po::value<fs::path>(&trainingset)->required()->default_value("data/ibug_lfpw_trainset"),
				"path to ibug LFPW example images and landmarks")
			("mean,m", po::value<fs::path>(&meanfile)->required()->default_value("data/mean_ibug_lfpw_68.txt"),
				"pre-calculated mean from ibug LFPW")
			("config,c", po::value<fs::path>(&configfile)->required(),
				"a model config file - specifies the training parameters of the model")
			("evaluation,e", po::value<fs::path>(&evaluationfile)->required(),
				"an evaluation config file - specifies how to evaluate the model (e.g. normalise by IED)")
			("output,o", po::value<fs::path>(&outputfile)->required()->default_value("model_tracking.txt"),
				"model output filename")
			("test-data,t", po::value<fs::path>(&testset)->required(),
				"path to ibug LFPW text images and landmarks")
			;
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: rcr-train-tracking [options]" << endl;
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
/*
	vector<Mat> loadedImages;
	vector<eos::core::LandmarkCollection<Vec2f>> loadedLandmarks;
	try	{
		std::tie(loadedImages, loadedLandmarks) = loadIbugData(trainingset);
	}
	catch (const fs::filesystem_error& ex)
	{
		cout << ex.what() << endl;
		return EXIT_FAILURE;
	}

	// Load the pre-calculated (and scaled) mean of all landmarks:
	Mat modelMean = loadMean(meanfile);
	
	// Read the information on which model landmarks to train on from the config:
	vector<string> modelLandmarks; // list read from the files, might be 68 or less
	try {
		modelLandmarks = readLandmarksListToTrain(configfile);
	}
	catch (const pt::ptree_error& error) {
		cout << "Error reading the training config: " << error.what() << endl;
		return EXIT_FAILURE;
	}
	catch (const std::logic_error& e) {
		cout << "Parsing config: " << e.what() << endl;
		return EXIT_FAILURE;
	}
	// Filter the landmarks:
	std::for_each(begin(loadedLandmarks), end(loadedLandmarks), [&modelLandmarks](eos::core::LandmarkCollection<Vec2f>& lm) { lm = eos::core::filter(lm, modelLandmarks); });
	// Reduce the mean:
	vector<string> ibugLandmarkIds; // full 68 point list
	for (int ibugId = 1; ibugId <= 68; ++ibugId) {
		ibugLandmarkIds.emplace_back(std::to_string(ibugId));
	}
	modelMean = rcr::toRow(eos::core::filter(rcr::toLandmarkCollection(modelMean, ibugLandmarkIds), modelLandmarks));

	// Read the evaluation information from the config:
	vector<string> rightEyeIdentifiers, leftEyeIdentifiers; // for ied calc. One or several.
	try {
		std::tie(rightEyeIdentifiers, leftEyeIdentifiers) =	readHowToCalculateTheIED(evaluationfile);
	}
	catch (const pt::ptree_error& error) {
		cout << "Error reading the evaluation config: " << error.what() << endl;
		return EXIT_FAILURE;
	}
	catch (const std::logic_error& e) {
		cout << "Parsing config: " << e.what() << endl;
		return EXIT_FAILURE;
	}

	// First, we detect the faces in the images and discard the false detections:
	vector<Mat> trainingImages;
	Mat x_gt, x_0;
	// If init with FDet and align to FB:
	// Each image, we add the original and perturb the fbox p times.
	// Augment the training set by perturbing the initialisations:
	auto perturb_t_mu = 0.0f; // in percent of IED or the face box? only needed in tracking?
	auto perturb_t_sigma = 0.02f; // in percent of the face box
	auto perturb_s_mu = 1.0f;
	auto perturb_s_sigma = 0.035f;
	// Do we need a flag: initialise from a) mean or b) GT? (tracking?)
	// a): only sigma makes sense. But I think ZF/SDM use also trans...
	// b): both makes sense in tracking! (we might set t=0)
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<> dist_t(perturb_t_mu, perturb_t_sigma); // TODO: If perturb_t_mu != 0, then we want plus and minus the value! handle this somehow.
	std::normal_distribution<> dist_s(perturb_s_mu, perturb_s_sigma);
	auto numPerturbations = 10; // = 2 perturbs + orig = 3 total
	{
		// Obtain the initial estimate x_0 using the ground truth and perturb it:
		for (size_t i = 0; i < loadedImages.size(); ++i) {
			// The initialisation values:
			// alignMean actually just transforms from [0.5...]-space to inside the facebox in img-coords.
			Mat x_0_curr = loadedLandmarks[i];
			x_0.push_back(rcr::alignMean(modelMean, cv::Rect(detectedFaces[0])));
			// Also copy the ground truth landmarks to one big data matrix:
			x_gt.push_back(rcr::toRow(loadedLandmarks[i]));
			// And the images:
			trainingImages.emplace_back(loadedImages[i]);
			for (auto p = 0; p < numPerturbations; ++p) {
				// Same again but create perturbations:
				Mat tmp = loadedImages[i].clone();
				cv::rectangle(tmp, detectedFaces[0], { 0.0f, 0.0f, 255.0f });
				cv::Rect tmp_pert = perturb(cv::Rect(detectedFaces[0]), dist_t(gen), dist_t(gen), dist_s(gen));
				// Note: FBox could be (partly) outside of image, but we check later during feature extraction, right?
				cv::rectangle(tmp, tmp_pert, { 255.0f, 0.0f, 0.0f });
				x_0.push_back(rcr::alignMean(modelMean, tmp_pert)); // tx, ty, s
				x_gt.push_back(rcr::toRow(loadedLandmarks[i]));
				trainingImages.emplace_back(loadedImages[i]);
			}
		}
	}
	// Else init with tracking and GT-LMs?
	cout << "Kept " << trainingImages.size() / (numPerturbations + 1) << " images out of " << loadedImages.size() << "." << endl;

	// Create 3 regularised linear regressors in series:
	vector<LinearRegressor<rcr::PartialPivLUSolveSolverDebug>> regressors;
	regressors.emplace_back(LinearRegressor<rcr::PartialPivLUSolveSolverDebug>(Regulariser(Regulariser::RegularisationType::MatrixNorm, 1.5f, false)));
	regressors.emplace_back(LinearRegressor<rcr::PartialPivLUSolveSolverDebug>(Regulariser(Regulariser::RegularisationType::MatrixNorm, 1.5f, false)));
	regressors.emplace_back(LinearRegressor<rcr::PartialPivLUSolveSolverDebug>(Regulariser(Regulariser::RegularisationType::MatrixNorm, 1.5f, false)));
	regressors.emplace_back(LinearRegressor<rcr::PartialPivLUSolveSolverDebug>(Regulariser(Regulariser::RegularisationType::MatrixNorm, 1.5f, false)));
	
	SupervisedDescentOptimiser<LinearRegressor<rcr::PartialPivLUSolveSolverDebug>, rcr::InterEyeDistanceNormalisation> supervisedDescentModel(regressors, rcr::InterEyeDistanceNormalisation(modelLandmarks, rightEyeIdentifiers, leftEyeIdentifiers));
	//SupervisedDescentOptimiser<LinearRegressor<PartialPivLUSolveSolverDebug>> supervisedDescentModel(regressors);
	
	std::vector<rcr::HoGParam> hog_params{ { VlHogVariant::VlHogVariantUoctti, 5, 11, 4, 55 }, { VlHogVariant::VlHogVariantUoctti, 5, 10, 4, 50 }, { VlHogVariant::VlHogVariantUoctti, 5, 8, 4, 40 },{ VlHogVariant::VlHogVariantUoctti, 5, 6, 4, 30 } }; // numCells, cellSize, numBins
	//std::vector<rcr::HoGParam> hog_params{ { VlHogVariant::VlHogVariantUoctti, 3, 5, 4, 15 },{ VlHogVariant::VlHogVariantUoctti, 3, 4, 4, 12 },{ VlHogVariant::VlHogVariantUoctti, 2, 5, 4, 10 } }; // numCells, cellSize, numBins
	assert(hog_params.size() == regressors.size());
	rcr::HogTransform hog(trainingImages, hog_params, modelLandmarks, rightEyeIdentifiers, leftEyeIdentifiers);

	// Train the model. We'll also specify an optional callback function:
	cout << "Training the model, printing the residual after each learned regressor: " << endl;
	// Rename to landmarkErrorCallback and put in the library?
	auto printResidual = [&x_gt, &modelLandmarks, &rightEyeIdentifiers, &leftEyeIdentifiers](const cv::Mat& currentPredictions) {
		cout << "NLSR train: " << cv::norm(currentPredictions, x_gt, cv::NORM_L2) / cv::norm(x_gt, cv::NORM_L2) << endl;
		
		Mat normalisedError = calculateNormalisedLandmarkErrors(currentPredictions, x_gt, modelLandmarks, rightEyeIdentifiers, leftEyeIdentifiers);
		cout << "Normalised LM-error train: " << cv::mean(normalisedError)[0] << endl;
	};
	
	supervisedDescentModel.train(x_gt, x_0, Mat(), hog, printResidual);

	// Save the learned model:
	rcr::detection_model learned_model(supervisedDescentModel, modelMean, modelLandmarks, hog_params, rightEyeIdentifiers, leftEyeIdentifiers);
	try {
		rcr::save_detection_model(learned_model, outputfile.string());
	}
	catch (cereal::Exception& e) {
		cout << e.what() << endl;
	}

	// ===========================
	// Evaluation on the test set:
	vector<Mat> loadedTestImages;
	vector<eos::core::LandmarkCollection<Vec2f>> loadedTestLandmarks;
	try	{
		std::tie(loadedTestImages, loadedTestLandmarks) = loadIbugData(testset);
	}
	catch (const fs::filesystem_error& ex)
	{
		cout << ex.what() << endl;
		return EXIT_FAILURE;
	}
	// Filter the landmarks:
	std::for_each(begin(loadedTestLandmarks), end(loadedTestLandmarks), [&modelLandmarks](eos::core::LandmarkCollection<Vec2f>& lm) { lm = eos::core::filter(lm, modelLandmarks); });

	// First, we detect the faces in the images and discard the false detections:
	vector<Mat> testImages;
	Mat x_ts_gt, x_ts_0;
	{
		// Load the face detector from OpenCV:
		cv::CascadeClassifier faceCascade;
		if (!faceCascade.load(facedetector.string()))
		{
			cout << "Error loading face detection model." << endl;
			return EXIT_FAILURE;
		}

		// Run the face detector and obtain the initial estimate x_0 using the mean landmarks:
		for (size_t i = 0; i < loadedTestImages.size(); ++i) {
			vector<cv::Rect> detectedFaces;
			faceCascade.detectMultiScale(loadedTestImages[i], detectedFaces, 1.2, 2, 0, cv::Size(50, 50));
			// Verify that the detected face is not a false positive:
			bool isTruePositive = rcr::checkFace(detectedFaces, loadedTestLandmarks[i]);
			if (isTruePositive) {
				// The initialisation values:
				x_ts_0.push_back(rcr::alignMean(modelMean, cv::Rect(detectedFaces[0])));
				// Also copy the ground truth landmarks to one big data matrix:
				x_ts_gt.push_back(rcr::toRow(loadedTestLandmarks[i]));
				// And the images:
				testImages.emplace_back(loadedTestImages[i]);
			}
		}
	}
	cout << "Kept " << testImages.size() << " images out of " << loadedTestImages.size() << "." << endl;

	Mat normalisedErrorInit = calculateNormalisedLandmarkErrors(x_ts_0, x_ts_gt, modelLandmarks, rightEyeIdentifiers, leftEyeIdentifiers);
	cout << "Normalised LM-error test from mean init: " << cv::mean(normalisedErrorInit)[0] << endl;
	
	Mat result = supervisedDescentModel.test(x_ts_0, Mat(), rcr::HogTransform(testImages, hog_params, modelLandmarks, rightEyeIdentifiers, leftEyeIdentifiers));
	
	cout << "NLSR test: " << cv::norm(result, x_ts_gt, cv::NORM_L2) / cv::norm(x_ts_gt, cv::NORM_L2) << endl;
	Mat normalisedError = calculateNormalisedLandmarkErrors(result, x_ts_gt, modelLandmarks, rightEyeIdentifiers, leftEyeIdentifiers);
	cout << "Normalised LM-error test: " << cv::mean(normalisedError)[0] << endl;
	
	Mat perLandmarkError;
	cv::reduce(normalisedError, perLandmarkError, 0, CV_REDUCE_AVG); // reduce to one row
	
	outputfile.replace_extension("error.txt");
	std::ofstream file(outputfile.string());
	for (int i = 0; i < perLandmarkError.cols; ++i) {
		file << perLandmarkError.at<float>(i);
		if (i != perLandmarkError.cols - 1) {
			file << ", ";
		}
	}
	file << std::endl;
*/
	return EXIT_SUCCESS;
}
