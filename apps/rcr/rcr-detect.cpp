/*
 * superviseddescent: A C++11 implementation of the supervised descent
 *                    optimisation method
 * File: apps/rcr/rcr-detect.cpp
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

#include <vector>
#include <iostream>
#include <fstream>

using namespace superviseddescent;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using std::vector;
using std::cout;
using std::endl;

/**
 * This app demonstrates the robust cascaded regression landmark detection from
 * "Random Cascaded-Regression Copse for Robust Facial Landmark Detection", 
 * Z. Feng, P. Huber, J. Kittler, W. Christmas, X.J. Wu,
 * IEEE Signal Processing Letters, Vol:22(1), 2015.
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
			("model,m", po::value<fs::path>(&modelfile)->required()->default_value("data/rcr/face_landmarks_model_rcr_22.bin"),
				"learned landmark detection model")
			("image,i", po::value<fs::path>(&inputimage)->required(),
				"input image file")
			("output,o", po::value<fs::path>(&outputfile)->required()->default_value("out.png"),
				"filename for the result image")
			;
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: rcr-detect [options]" << endl;
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

	cv::Mat image = cv::imread(inputimage.string());

	// Run the face detector:
	vector<cv::Rect> detected_faces;
	face_cascade.detectMultiScale(image, detected_faces, 1.2, 2, 0, cv::Size(50, 50));
	if (detected_faces.empty())	{
		cout << "No face detected. Exiting." << endl;
		return EXIT_SUCCESS;
	}
	
	// Detect the landmarks:
	auto landmarks = rcr_model.detect(image, detected_faces[0]);

	rcr::draw_landmarks(image, landmarks);
	cv::imwrite(outputfile.string(), image);

	return EXIT_SUCCESS;
}
