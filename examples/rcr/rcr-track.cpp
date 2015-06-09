/*
 * superviseddescent: A C++11 implementation of the supervised descent
 *                    optimisation method
 * File: examples/rcr/rcr-track.cpp
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

#include "eos_core_landmark.hpp"
#include "eos_io_landmarks.hpp"
#include "helpers.hpp"
#include "model.hpp"

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

#include <vector>
#include <iostream>
#include <fstream>

using namespace superviseddescent;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using cv::Mat;
using std::vector;
using std::cout;
using std::endl;

template<class T = int>
cv::Rect_<T> get_enclosing_bbox(cv::Mat landmarks)
{
	auto num_landmarks = landmarks.cols / 2;
	double min_x_val, max_x_val, min_y_val, max_y_val;
	cv::minMaxLoc(landmarks.colRange(0, num_landmarks), &min_x_val, &max_x_val);
	cv::minMaxLoc(landmarks.colRange(num_landmarks, landmarks.cols), &min_y_val, &max_y_val);
	return cv::Rect_<T>(min_x_val, min_y_val, max_x_val - min_x_val, max_y_val - min_y_val);
};

/**
 * This app builds upon the robust cascaded regression landmark detection from
 * "Random Cascaded-Regression Copse for Robust Facial Landmark Detection", 
 * Z. Feng, P. Huber, J. Kittler, W. Christmas, X.J. Wu,
 * IEEE Signal Processing Letters, Vol:22(1), 2015.
 * It modifies the approach to track a face in a video.
 *
 * It loads a model trained with rcr-train, detects a face using OpenCV's face
 * detector, and then runs the landmark detection.
 */
int main(int argc, char *argv[])
{
	fs::path facedetector, inputimage, modelfile, outputfile;
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"display the help message")
			("facedetector,f", po::value<fs::path>(&facedetector)->required(),
				"full path to OpenCV's face detector (haarcascade_frontalface_alt2.xml)")
			("model,m", po::value<fs::path>(&modelfile)->required()->default_value("../data/rcr/face_landmarks_model_rcr_tracking_29.txt"),
				"learned landmark detection model")
			("image,i", po::value<fs::path>(&inputimage),
				"input video file. If not specified, the camera will be used.")
			("output,o", po::value<fs::path>(&outputfile)->required()->default_value("out.png"),
				"filename for the result image")
			;
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: rcr-track [options]" << endl;
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

	rcr::detection_model rcr_model;

	// Load the learned model:
	try {
		rcr_model = rcr::load_detection_model(modelfile.string());
	}
	catch (const cereal::Exception& e) {
		cout << "Error reading the RCR model " << modelfile << ": " << e.what() << endl;
		return EXIT_FAILURE;
	}
	
	// Load the face detector from OpenCV:
	cv::CascadeClassifier face_cascade;
	if (!face_cascade.load(facedetector.string()))
	{
		cout << "Error loading the face detector " << facedetector << "." << endl;
		return EXIT_FAILURE;
	}

	cv::VideoCapture cap;
	if (inputimage.empty()) {
		cap.open(0); // no file given, open the default camera
	}
	else {
		cap.open(inputimage.string());
	}
	if (!cap.isOpened()) {  // check if we succeeded
		cout << "Couldn't open the given file or camera 0." << endl;
		return EXIT_FAILURE;
	}

	cv::namedWindow("video", 1);
	Mat image;
	using namespace std::chrono;
	time_point<system_clock> start, end;

	bool have_face = false;
	eos::core::LandmarkCollection<cv::Vec2f> current_landmarks;

	for (;;)
	{
		cap >> image; // get a new frame from camera
		
		if (!have_face) {
			// Run the face detector and obtain the initial estimate using the mean landmarks:
			start = system_clock::now();
			vector<cv::Rect> detected_faces;
			face_cascade.detectMultiScale(image, detected_faces, 1.1, 3, 0, cv::Size(110, 110));
			end = system_clock::now();
			auto t_fd = duration_cast<milliseconds>(end - start).count();
			if (detected_faces.empty()) {
				cv::imshow("video", image);
				cv::waitKey(30);
				continue;
			}
			cv::rectangle(image, detected_faces[0], { 255, 0, 0 });
			
			start = system_clock::now();
			current_landmarks = rcr_model.detect(image, detected_faces[0]);
			end = system_clock::now();
			auto t_fit = duration_cast<milliseconds>(end - start).count();

			rcr::draw_landmarks(image, current_landmarks);
			have_face = true;

			cout << "FD:" << t_fd << "\tLM: " << t_fit << endl;
		}
		else {
			// We already have a face. Do Zhenhua's magic:
			// current_landmarks are the estimate from the last frame
			rcr::draw_landmarks(image, current_landmarks, { 0, 0, 255 }); // red, from last frame
			auto enclosing_bbox = get_enclosing_bbox(rcr::to_row(current_landmarks));
			cv::rectangle(image, enclosing_bbox, { 0, 0, 255 });
			// we need to make the enclosing bbox square again (or correct with 1/... in align_mean)
			float make_bbox_square_factor = std::max(enclosing_bbox.width, enclosing_bbox.height) / static_cast<float>(std::min(enclosing_bbox.width, enclosing_bbox.height));
			if (enclosing_bbox.width > enclosing_bbox.height) {
				auto diff = enclosing_bbox.width - enclosing_bbox.height;
				enclosing_bbox.y -= diff / 2.0f;
				enclosing_bbox.height = enclosing_bbox.width;
			}
			else {
				auto diff = enclosing_bbox.height - enclosing_bbox.width;
				enclosing_bbox.x -= diff / 2.0f;
				enclosing_bbox.width = enclosing_bbox.height;
			}
			cv::rectangle(image, enclosing_bbox, { 0, 127, 255 });
			float width_of_mean = get_enclosing_bbox<float>(rcr_model.get_mean()).width;
			float height_of_mean = get_enclosing_bbox<float>(rcr_model.get_mean()).height;
			float scaling = 1 / std::max(width_of_mean, height_of_mean); // so the model will be scaled with constant aspect ratio
			auto aligned_mean = rcr::align_mean(rcr_model.get_mean(), enclosing_bbox, scaling, scaling, 0.0f, -0.33f);
			rcr::draw_landmarks(image, aligned_mean, { 255, 0, 0 }); // blue, mean init from enclosing_bbox

			current_landmarks = rcr_model.detect(image, aligned_mean);
			rcr::draw_landmarks(image, current_landmarks, { 0, 255, 0 }); // green, the new optimised landmarks
			// have some condition to set have_face = false
		}

		cv::imshow("video", image);
		if (cv::waitKey(30) >= 0) break;
	}

	return EXIT_SUCCESS;
}
