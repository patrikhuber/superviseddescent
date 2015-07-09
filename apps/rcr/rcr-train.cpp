/*
 * superviseddescent: A C++11 implementation of the supervised descent
 *                    optimisation method
 * File: apps/rcr/rcr-train.cpp
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
#include "superviseddescent/verbose_solver.hpp"

#include "rcr/landmarks_io.hpp"
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
std::pair<vector<Mat>, vector<rcr::LandmarkCollection<Vec2f>>> load_ibug_data(fs::path directory)
{
	vector<Mat> images;
	vector<rcr::LandmarkCollection<Vec2f>> landmarks;

	// Get all the filenames in the given directory:
	vector<fs::path> image_filenames;
	fs::directory_iterator end_itr;
	for (fs::directory_iterator i(directory); i != end_itr; ++i)
	{
		if (fs::is_regular_file(i->status()) && i->path().extension() == ".png")
			image_filenames.emplace_back(i->path());
	}

	// Load each image and its landmarks into memory:
	for (auto file : image_filenames)
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
cv::Mat load_mean(fs::path filename)
{
	std::ifstream file(filename.string());
	if (!file.is_open()) {
		throw std::runtime_error(string("Could not open file: " + filename.string()));
	}

	string line;
	getline(file, line);

	vector<string> values;
	boost::split(values, line, boost::is_any_of(","));

	int twice_num_landmarks = static_cast<int>(values.size());
	Mat mean(1, twice_num_landmarks, CV_32FC1);
	for (int i = 0; i < twice_num_landmarks; ++i) {
		mean.at<float>(i) = std::stof(values[i]);
	}

	return mean;
};

/**
 * Perturb by a certain x and y translation and an optional scaling.
 *
 * tx, ty are in percent of the total face box width/height.
 *
 * @param[in] facebox A facebox to align the model to.
 * @param[in] translation_x Translation in x of the box.
 * @param[in] translation_y Translation in y of the box.
 * @param[in] scaling Optional scale factor of the box.
 * @return A perturbed cv::Rect.
 */
cv::Rect perturb(cv::Rect facebox, float translation_x, float translation_y, float scaling = 1.0f)
{
	
	auto tx_pixel = translation_x * facebox.width;
	auto ty_pixel = translation_y * facebox.height;
	// Because the reference point is on the top left and not in the center, we
	// need to temporarily store the new width and calculate half of the offset.
	// We need to move it further to compensate for the scaling, i.e. keep the center the center.
	auto perturbed_width = facebox.width * scaling;
	auto perturbed_height = facebox.height * scaling;
	//auto perturbed_width_diff = facebox.width - perturbed_width;
	//auto perturbed_height_diff = facebox.height - perturbed_height;
	// Note: Rounding?
	cv::Rect perturbed_box(facebox.x + (facebox.width - perturbed_width) / 2.0f + tx_pixel, facebox.y + (facebox.height - perturbed_height) / 2.0f + ty_pixel, perturbed_width, perturbed_height);

	return perturbed_box;
}

/**
 * Calculate the norm (L2 error, in pixel) of two landmarks.
 *
 * @param[in] prediction First landmark.
 * @param[in] groundtruth Second landmark.
 * @return The L2 norm of the two landmark's coordinates.
 */
double norm(const rcr::Landmark<Vec2f>& prediction, const rcr::Landmark<Vec2f>& groundtruth)
{
	return cv::norm(prediction.coordinates, groundtruth.coordinates, cv::NORM_L2);
};

/**
 * Calculate the element-wise L2 norm of two sets of landmarks.
 *
 * Requires both LandmarkCollections to have the same size.
 *
 * @param[in] prediction First set of landmarks.
 * @param[in] groundtruth Second set of landmarks.
 * @return A row-vector with each entry being the L2 norm of the two respective landmarks.
 */
Mat elementwise_norm(const rcr::LandmarkCollection<Vec2f>& prediction, const rcr::LandmarkCollection<Vec2f>& groundtruth)
{
	assert(prediction.size() == groundtruth.size());
	Mat result(1, prediction.size(), CV_32FC1); // a row with each entry a norm
	for (std::size_t i = 0; i < prediction.size(); ++i) {
		result.at<float>(i) = norm(prediction[i], groundtruth[i]);
	}
	return result;
};

/**
 * Calculate the element-wise L2 norm of two sets of landmark
 * collections, normalised by the inter eye distance of the
 * predictions. Each row in the given matrices should correspond
 * to one LandmarkCollection.
 *
 * \c [right|left]_eye_identifiers are used to calculate the inter eye distance.
 *
 * Requires both LandmarkCollections to have the same size.
 *
 * Note:
 *  double mean_error = cv::mean(normalised_errors)[0]; // = the mean over all, identical to simple case
 *  cv::reduce(normalised_errors, normalised_errors, 0, CV_REDUCE_AVG); // reduce to one row
 *
 * @param[in] prediction First set of landmarks.
 * @param[in] groundtruth Second set of landmarks.
 * @param[in] model_landmarks A mapping from indices to landmark identifiers.
 * @param[in] right_eye_identifiers A list of landmark identifiers that specifies which landmarks make up the right eye.
 * @param[in] left_eye_identifiers A list of landmark identifiers that specifies which landmarks make up the left eye.
 * @return A matrix where each row corresponds to a separate set of landmarks (i.e. a different image), and each column is a different landmark's L2 norm.
 */
Mat calculate_normalised_landmark_errors(Mat predictions, Mat groundtruth, vector<string> model_landmarks, vector<string> right_eye_identifiers, vector<string> left_eye_identifiers)
{
	assert(predictions.rows == groundtruth.rows && predictions.cols == groundtruth.cols);
	Mat normalised_errors;
	for (int r = 0; r < predictions.rows; ++r) {
		auto pred = rcr::to_landmark_collection(predictions.row(r), model_landmarks);
		auto gt = rcr::to_landmark_collection(groundtruth.row(r), model_landmarks);
		// calculates the element-wise norm, normalised with the IED:
		Mat landmark_norms = elementwise_norm(pred, gt).mul(1.0f / rcr::get_ied(pred, right_eye_identifiers, left_eye_identifiers));
		normalised_errors.push_back(landmark_norms);
	}
	return normalised_errors;
};

/**
 * Reads a list of which landmarks to train.
 *
 * @param[in] configfile A training config file to read.
 * @return A list that contains all the landmark identifiers from the config that are to be used for training.
 */
vector<string> read_landmarks_list_to_train(fs::path configfile)
{
	pt::ptree config_tree;
	pt::read_info(configfile.string(), config_tree); // Can throw a pt::ptree_error, maybe try/catch
	
	vector<string> model_landmarks;
	// Get stuff from the modelLandmarks subtree:
	pt::ptree pt_model_landmarks = config_tree.get_child("modelLandmarks");
	string model_landmarks_usage = pt_model_landmarks.get<string>("landmarks");
	if (model_landmarks_usage.empty()) {
		// value is empty, meaning it's a node and the user should specify a list of 'landmarks'
		pt::ptree pt_model_landmarks_list = pt_model_landmarks.get_child("landmarks");
		for (const auto& kv : pt_model_landmarks_list) {
			model_landmarks.push_back(kv.first);
		}
		cout << "Loaded a list of " << model_landmarks.size() << " landmarks to train the model." << endl;
	}
	else if (model_landmarks_usage == "all") {
		throw std::logic_error("Using 'all' modelLandmarks is not implemented yet - specify a list for now.");
	}
	else {
		throw std::logic_error("Error reading the models 'landmarks' key, should either provide a node with a list of landmarks or specify 'all'.");
	}
	return model_landmarks;
}

/**
 * Reads a config file ('eval.txt') that specifies which landmarks make
 * up the eyes and are to be used to calculate the IED.
 *
 * @param[in] evaluationfile A training config file to read.
 * @return A pair with the right and left eye identifiers.
 * @throws A ptree or logic error?
 */
std::pair<vector<string>, vector<string>> read_how_to_calculate_the_IED(fs::path evaluationfile)
{
	vector<string> right_eye_identifiers, left_eye_identifiers;

	pt::ptree eval_config_tree;
	string right_eye;
	string left_eye;
	pt::read_info(evaluationfile.string(), eval_config_tree); // could throw a boost::property_tree::ptree_error, maybe try/catch

	pt::ptree pt_parameters = eval_config_tree.get_child("interEyeDistance");
	right_eye = pt_parameters.get<string>("rightEye");
	left_eye = pt_parameters.get<string>("leftEye");
	
	// Process the interEyeDistance landmarks - one or two identifiers might be given
	boost::split(right_eye_identifiers, right_eye, boost::is_any_of(" "));
	boost::split(left_eye_identifiers, left_eye, boost::is_any_of(" "));
	return std::make_pair(right_eye_identifiers, left_eye_identifiers);
}

/**
 * This app implements the training of a robust cascaded regression landmark
 * detection model, similar to the one proposed in the paper:
 * "Random Cascaded-Regression Copse for Robust Facial Landmark Detection",
 * Z. Feng, P. Huber, J. Kittler, W. Christmas, X.J. Wu,
 * IEEE Signal Processing Letters, Vol:22(1), 2015.
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
	fs::path trainingset, meanfile, facedetector, configfile, evaluationfile, outputfile, testset;
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"display the help message")
			("data,d", po::value<fs::path>(&trainingset)->required(),
				"path to ibug LFPW example images and landmarks")
			("mean,m", po::value<fs::path>(&meanfile)->required()->default_value("data/rcr/mean_ibug_lfpw_68.txt"),
				"pre-calculated mean from ibug LFPW")
			("facedetector,f", po::value<fs::path>(&facedetector)->required(),
				"full path to OpenCV's face detector (haarcascade_frontalface_alt2.xml)")
			("config,c", po::value<fs::path>(&configfile)->required()->default_value("data/rcr/rcr_training_22.cfg"),
				"a model config file - specifies the training parameters of the model")
			("evaluation,e", po::value<fs::path>(&evaluationfile)->required()->default_value("data/rcr/rcr_eval.cfg"),
				"an evaluation config file - specifies how to evaluate the model (e.g. normalise by IED)")
			("output,o", po::value<fs::path>(&outputfile)->required()->default_value("model.bin"),
				"model output filename")
			("test-data,t", po::value<fs::path>(&testset)->required(),
				"path to ibug LFPW test images and landmarks")
			;
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: rcr-train [options]" << endl;
			cout << desc;
			return EXIT_SUCCESS;
		}
		po::notify(vm);
	}
	catch (const po::error& e) {
		cout << "Error while parsing command-line arguments: " << e.what() << endl;
		cout << "Use --help to display a list of options." << endl;
		return EXIT_SUCCESS;
	}

	vector<Mat> loaded_images;
	vector<rcr::LandmarkCollection<Vec2f>> loaded_landmarks;
	try	{
		std::tie(loaded_images, loaded_landmarks) = load_ibug_data(trainingset);
	}
	catch (const fs::filesystem_error& e)
	{
		cout << e.what() << endl;
		return EXIT_FAILURE;
	}

	// Load the pre-calculated (and scaled) mean of all landmarks:
	Mat model_mean = load_mean(meanfile);
	
	// Read the information on which model landmarks to train on from the config:
	vector<string> model_landmarks; // list read from the files, might be 68 or less
	try {
		model_landmarks = read_landmarks_list_to_train(configfile);
	}
	catch (const pt::ptree_error& e) {
		cout << "Error reading the training config: " << e.what() << endl;
		return EXIT_FAILURE;
	}
	catch (const std::logic_error& e) {
		cout << "Parsing config: " << e.what() << endl;
		return EXIT_FAILURE;
	}
	// Filter the landmarks:
	std::for_each(begin(loaded_landmarks), end(loaded_landmarks), [&model_landmarks](rcr::LandmarkCollection<Vec2f>& lm) { lm = rcr::filter(lm, model_landmarks); });
	// Reduce the mean:
	vector<string> ibug_landmark_ids; // full 68 point list
	for (int ibug_id = 1; ibug_id <= 68; ++ibug_id) {
		ibug_landmark_ids.emplace_back(std::to_string(ibug_id));
	}
	model_mean = rcr::to_row(rcr::filter(rcr::to_landmark_collection(model_mean, ibug_landmark_ids), model_landmarks));

	// Read the evaluation information from the config:
	vector<string> right_eye_identifiers, left_eye_identifiers; // for ied calc. One or several.
	try {
		std::tie(right_eye_identifiers, left_eye_identifiers) =	read_how_to_calculate_the_IED(evaluationfile);
	}
	catch (const pt::ptree_error& e) {
		cout << "Error reading the evaluation config: " << e.what() << endl;
		return EXIT_FAILURE;
	}
	catch (const std::logic_error& e) {
		cout << "Parsing config: " << e.what() << endl;
		return EXIT_FAILURE;
	}

	// First, we detect the faces in the images and discard the false detections:
	vector<Mat> training_images;
	Mat x_gt, x_0;
	// Augment the training set by perturbing the initialisations:
	auto perturb_t_mu = 0.0f; // in percent of IED or the face box? only needed in tracking?
	auto perturb_t_sigma = 0.04f; // in percent of the face box
	auto perturb_s_mu = 1.0f;
	auto perturb_s_sigma = 0.04f;
	// If we initialise from the mean, I thought only t_sigma makes sense and t_mu should be zero. But I think ZF/SDM use also t_mu.
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<> dist_t(perturb_t_mu, perturb_t_sigma); // Todo: If perturb_t_mu != 0, then we want plus and minus the value! handle this somehow.
	std::normal_distribution<> dist_s(perturb_s_mu, perturb_s_sigma);
	// Each image, we add the original and perturb the face box num_perturbations times.
	auto num_perturbations = 10; // = 10 perturbs + orig = 11 total
	{
		// Load the face detector from OpenCV:
		cv::CascadeClassifier face_cascade;
		if (!face_cascade.load(facedetector.string()))
		{
			cout << "Error loading face detection model." << endl;
			return EXIT_FAILURE;
		}
	
		// Run the face detector and obtain the initial estimate x_0 using the mean landmarks:
		for (size_t i = 0; i < loaded_images.size(); ++i) {
			vector<cv::Rect> detected_faces;
			face_cascade.detectMultiScale(loaded_images[i], detected_faces, 1.2, 2, 0, cv::Size(50, 50));
			// Verify that the detected face is not a false positive:
			bool is_true_positive = rcr::check_face(detected_faces, loaded_landmarks[i]); // reduced landmarks. Meaning the check will vary.
			if (is_true_positive) {
				// The initialisation values:
				// alignMean actually just transforms from [0.5...]-space to inside the facebox in img-coords.
				x_0.push_back(rcr::align_mean(model_mean, cv::Rect(detected_faces[0])));
				// Also copy the ground truth landmarks to one big data matrix:
				x_gt.push_back(rcr::to_row(loaded_landmarks[i]));
				// And the images:
				training_images.emplace_back(loaded_images[i]);
				for (auto p = 0; p < num_perturbations; ++p) {
					// Same again but create perturbations:
					Mat tmp = loaded_images[i].clone();
					cv::rectangle(tmp, detected_faces[0], { 0.0f, 0.0f, 255.0f });
					cv::Rect tmp_pert = perturb(cv::Rect(detected_faces[0]), dist_t(gen), dist_t(gen), dist_s(gen));
					// Note: FBox could be (partly) outside of image, but we check later during feature extraction, right?
					cv::rectangle(tmp, tmp_pert, { 255.0f, 0.0f, 0.0f });
					x_0.push_back(rcr::align_mean(model_mean, tmp_pert)); // tx, ty, s
					x_gt.push_back(rcr::to_row(loaded_landmarks[i]));
					training_images.emplace_back(loaded_images[i]);
				}
			}
		}
	}
	// Else init with tracking and GT-LMs?
	cout << "Kept " << training_images.size() / (num_perturbations + 1) << " images out of " << loaded_images.size() << "." << endl;

	// Create 3 regularised linear regressors in series:
	vector<LinearRegressor<VerbosePartialPivLUSolver>> regressors;
	regressors.emplace_back(LinearRegressor<VerbosePartialPivLUSolver>(Regulariser(Regulariser::RegularisationType::MatrixNorm, 1.5f, false)));
	regressors.emplace_back(LinearRegressor<VerbosePartialPivLUSolver>(Regulariser(Regulariser::RegularisationType::MatrixNorm, 1.5f, false)));
	regressors.emplace_back(LinearRegressor<VerbosePartialPivLUSolver>(Regulariser(Regulariser::RegularisationType::MatrixNorm, 1.5f, false)));
	regressors.emplace_back(LinearRegressor<VerbosePartialPivLUSolver>(Regulariser(Regulariser::RegularisationType::MatrixNorm, 1.5f, false)));
	
	SupervisedDescentOptimiser<LinearRegressor<VerbosePartialPivLUSolver>, rcr::InterEyeDistanceNormalisation> supervised_descent_model(regressors, rcr::InterEyeDistanceNormalisation(model_landmarks, right_eye_identifiers, left_eye_identifiers));
	
	std::vector<rcr::HoGParam> hog_params{ { VlHogVariant::VlHogVariantUoctti, 5, 11, 4, 1.0f },{ VlHogVariant::VlHogVariantUoctti, 5, 10, 4, 0.7f },{ VlHogVariant::VlHogVariantUoctti, 5, 8, 4, 0.4f },{ VlHogVariant::VlHogVariantUoctti, 5, 6, 4, 0.25f } }; // 3 /*numCells*/, 12 /*cellSize*/, 4 /*numBins*/
	assert(hog_params.size() == regressors.size());
	rcr::HogTransform hog(training_images, hog_params, model_landmarks, right_eye_identifiers, left_eye_identifiers);

	// Train the model. We'll also specify an optional callback function:
	cout << "Training the model, printing the residual after each learned regressor: " << endl;
	// Note: Rename to landmark_error_callback and put in the library?
	auto print_residual = [&x_gt, &model_landmarks, &right_eye_identifiers, &left_eye_identifiers](const cv::Mat& current_predictions) {
		cout << "NLSR train: " << cv::norm(current_predictions, x_gt, cv::NORM_L2) / cv::norm(x_gt, cv::NORM_L2) << endl;
		
		Mat normalised_error = calculate_normalised_landmark_errors(current_predictions, x_gt, model_landmarks, right_eye_identifiers, left_eye_identifiers);
		cout << "Normalised LM-error train: " << cv::mean(normalised_error)[0] << endl;
	};
	
	supervised_descent_model.train(x_gt, x_0, Mat(), hog, print_residual);

	// Save the learned model:
	rcr::detection_model learned_model(supervised_descent_model, model_mean, model_landmarks, hog_params, right_eye_identifiers, left_eye_identifiers);
	try {
		rcr::save_detection_model(learned_model, outputfile.string());
	}
	catch (const cereal::Exception& e) {
		cout << e.what() << endl;
	}

	// ===========================
	// Evaluation on the test set:
	vector<Mat> loaded_test_images;
	vector<rcr::LandmarkCollection<Vec2f>> loaded_test_landmarks;
	try	{
		std::tie(loaded_test_images, loaded_test_landmarks) = load_ibug_data(testset);
	}
	catch (const fs::filesystem_error& e)
	{
		cout << e.what() << endl;
		return EXIT_FAILURE;
	}
	// Filter the landmarks:
	std::for_each(begin(loaded_test_landmarks), end(loaded_test_landmarks), [&model_landmarks](rcr::LandmarkCollection<Vec2f>& lm) { lm = rcr::filter(lm, model_landmarks); });

	// First, we detect the faces in the images and discard the false detections:
	vector<Mat> test_images;
	Mat x_ts_gt, x_ts_0;
	{
		// Load the face detector from OpenCV:
		cv::CascadeClassifier face_cascade;
		if (!face_cascade.load(facedetector.string()))
		{
			cout << "Error loading face detection model." << endl;
			return EXIT_FAILURE;
		}

		// Run the face detector and obtain the initial estimate x_0 using the mean landmarks:
		for (size_t i = 0; i < loaded_test_images.size(); ++i) {
			vector<cv::Rect> detected_faces;
			face_cascade.detectMultiScale(loaded_test_images[i], detected_faces, 1.2, 2, 0, cv::Size(50, 50));
			// Verify that the detected face is not a false positive:
			bool is_true_positive = rcr::check_face(detected_faces, loaded_test_landmarks[i]);
			if (is_true_positive) {
				// The initialisation values:
				x_ts_0.push_back(rcr::align_mean(model_mean, cv::Rect(detected_faces[0])));
				// Also copy the ground truth landmarks to one big data matrix:
				x_ts_gt.push_back(rcr::to_row(loaded_test_landmarks[i]));
				// And the images:
				test_images.emplace_back(loaded_test_images[i]);
			}
		}
	}
	cout << "Kept " << test_images.size() << " images out of " << loaded_test_images.size() << "." << endl;

	Mat normalised_error_init = calculate_normalised_landmark_errors(x_ts_0, x_ts_gt, model_landmarks, right_eye_identifiers, left_eye_identifiers);
	cout << "Normalised LM-error test from mean init: " << cv::mean(normalised_error_init)[0] << endl;
	
	Mat result = supervised_descent_model.test(x_ts_0, Mat(), rcr::HogTransform(test_images, hog_params, model_landmarks, right_eye_identifiers, left_eye_identifiers));
	
	cout << "NLSR test: " << cv::norm(result, x_ts_gt, cv::NORM_L2) / cv::norm(x_ts_gt, cv::NORM_L2) << endl;
	Mat normalised_error = calculate_normalised_landmark_errors(result, x_ts_gt, model_landmarks, right_eye_identifiers, left_eye_identifiers);
	cout << "Normalised LM-error test: " << cv::mean(normalised_error)[0] << endl;
	
	// We write out a file modelname.error.txt to be able to plot the per-landmark error:
	Mat per_landmark_error;
	cv::reduce(normalised_error, per_landmark_error, 0, CV_REDUCE_AVG); // reduce to one row
	
	outputfile.replace_extension("error.txt");
	std::ofstream file(outputfile.string());
	for (int i = 0; i < per_landmark_error.cols; ++i) {
		file << per_landmark_error.at<float>(i);
		if (i != per_landmark_error.cols - 1) {
			file << ", ";
		}
	}
	file << std::endl;

	return EXIT_SUCCESS;
}
