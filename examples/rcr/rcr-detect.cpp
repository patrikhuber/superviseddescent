/*
 * superviseddescent: A C++11 implementation of the supervised descent
 *                    optimisation method
 * File: examples/rcr/rcr-detect.cpp
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
#include "adaptive_vlhog.hpp"
#include "helpers.hpp"
#include "model.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"
#include "boost/archive/text_iarchive.hpp"
#include "boost/serialization/string.hpp"

#include <vector>
#include <iostream>
#include <fstream>
#include <utility>
#include <random>
#include <cassert>

using namespace superviseddescent;
using namespace eos;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using cv::Mat;
using cv::Vec2f;
using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::size_t;

// Duplicate from rcr-train.cpp, not needed here at all!
class PartialPivLUSolveSolverDebug
{
public:
	// Note: we should leave the choice of inverting A or AtA to the solver.
	// But this also means we need to pass through the regularisation params.
	// We can't just pass a cv::Mat regularisation because the dimensions for
	// regularising A and AtA are different.
	cv::Mat solve(cv::Mat data, cv::Mat labels, Regulariser regulariser)
	{
		using cv::Mat;
		using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
		using namespace std::chrono;
		time_point<system_clock> start, end;

		// Map the cv::Mat data and labels to Eigen matrices:
		Eigen::Map<RowMajorMatrixXf> A_Eigen(data.ptr<float>(), data.rows, data.cols);
		Eigen::Map<RowMajorMatrixXf> labels_Eigen(labels.ptr<float>(), labels.rows, labels.cols);

		start = system_clock::now();
		RowMajorMatrixXf AtA_Eigen = A_Eigen.transpose() * A_Eigen;
		end = system_clock::now();
		cout << "At * A (ms): " << duration_cast<milliseconds>(end - start).count() << endl;

		// Note: This is a bit of unnecessary back-and-forth mapping, just for the regularisation:
		Mat AtA_Map(static_cast<int>(AtA_Eigen.rows()), static_cast<int>(AtA_Eigen.cols()), CV_32FC1, AtA_Eigen.data());
		Mat regularisationMatrix = regulariser.getMatrix(AtA_Map, data.rows);
		Eigen::Map<RowMajorMatrixXf> reg_Eigen(regularisationMatrix.ptr<float>(), regularisationMatrix.rows, regularisationMatrix.cols);

		Eigen::DiagonalMatrix<float, Eigen::Dynamic> reg_Eigen_diag(regularisationMatrix.rows);
		Eigen::VectorXf diagVec(regularisationMatrix.rows);
		for (int i = 0; i < diagVec.size(); ++i) {
			diagVec(i) = regularisationMatrix.at<float>(i, i);
		}
		reg_Eigen_diag.diagonal() = diagVec;
		start = system_clock::now();
		AtA_Eigen = AtA_Eigen + reg_Eigen_diag.toDenseMatrix();
		end = system_clock::now();
		cout << "AtA + Reg (ms): " << duration_cast<milliseconds>(end - start).count() << endl;

		// Perform a PartialPivLU:
		start = system_clock::now();
		Eigen::PartialPivLU<RowMajorMatrixXf> qrOfAtA(AtA_Eigen);
		end = system_clock::now();
		cout << "Decomposition (ms): " << duration_cast<milliseconds>(end - start).count() << endl;
		start = system_clock::now();
		//RowMajorMatrixXf AtAInv_Eigen = qrOfAtA.inverse();
		RowMajorMatrixXf x_Eigen = qrOfAtA.solve(A_Eigen.transpose() * labels_Eigen);
		//RowMajorMatrixXf x_Eigen = AtA_Eigen.partialPivLu.solve(A_Eigen.transpose() * labels_Eigen);
		end = system_clock::now();
		cout << "solve() (ms): " << duration_cast<milliseconds>(end - start).count() << endl;

		// x = (AtAReg)^-1 * At * b:
		start = system_clock::now();
		//RowMajorMatrixXf x_Eigen = AtAInv_Eigen * A_Eigen.transpose() * labels_Eigen;
		end = system_clock::now();
		cout << "AtAInv * At * b (ms): " << duration_cast<milliseconds>(end - start).count() << endl;

		// Map the resulting x back to a cv::Mat by creating a Mat header:
		Mat x(static_cast<int>(x_Eigen.rows()), static_cast<int>(x_Eigen.cols()), CV_32FC1, x_Eigen.data());

		// We have to copy the data because the underlying data is managed by Eigen::Matrix x_Eigen, which will go out of scope after we leave this function:
		return x.clone();
		//return qrOfAtA.isInvertible();
	};
};

// Duplicate from rcr-train.cpp
class RCRModel
{
public:
	SupervisedDescentOptimiser<LinearRegressor<PartialPivLUSolveSolverDebug>, rcr::InterEyeDistanceNormalisation> sdm_model;
	std::vector<rcr::HoGParam> hog_params;	///< The hog parameters for each regressor level
	vector<string> modelLandmarks;			///< The landmark identifiers the model consists of
	vector<string> rightEyeIdentifiers, leftEyeIdentifiers;	///< Holds information about which landmarks are the eyes, to calculate the IED normalisation for the adaptive update
	cv::Mat model_mean;						///< The mean of the model, learned and scaled from training data, given a specific face detector

private:
	friend class boost::serialization::access;
	/**
	 * Serialises this class using boost::serialization.
	 *
	 * @param[in] ar The archive to serialise to (or to serialise from).
	 * @param[in] version An optional version argument.
	 */
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /*version*/)
	{
		ar & sdm_model & hog_params & modelLandmarks & rightEyeIdentifiers & leftEyeIdentifiers & model_mean;
	}
};

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
			("model,m", po::value<fs::path>(&modelfile)->required()->default_value("data/rcr/face_landmarks_model_rcr_29.txt"),
				"learned landmark detection model")
			("image,i", po::value<fs::path>(&inputimage)->required()->default_value("data/ibug_lfpw_trainset/bla.png"),
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
	catch (po::error& e) {
		cout << "Error while parsing command-line arguments: " << e.what() << endl;
		cout << "Use --help to display a list of options." << endl;
		return EXIT_SUCCESS;
	}

	RCRModel rcr_model;
	// Load the learned model:
	{
		std::ifstream learnedModelFile(modelfile.string());
		boost::archive::text_iarchive ia(learnedModelFile);
		ia >> rcr_model;
	}
	
	// Load the face detector from OpenCV:
	cv::CascadeClassifier faceCascade;
	if (!faceCascade.load(facedetector.string()))
	{
		cout << "Error loading face detection model." << endl;
		return EXIT_FAILURE;
	}

	cv::Mat image = cv::imread(inputimage.string());

	// Run the face detector and obtain the initial estimate using the mean landmarks:
	vector<cv::Rect> detectedFaces;
	faceCascade.detectMultiScale(image, detectedFaces, 1.2, 2, 0, cv::Size(50, 50));
	
	cv::Mat init = rcr::alignMean(rcr_model.model_mean, cv::Rect(detectedFaces[0]));
	rcr::drawLandmarks(image, init, {0, 0, 255});
	rcr::HogTransform hog({ image }, rcr_model.hog_params, rcr_model.modelLandmarks, rcr_model.rightEyeIdentifiers, rcr_model.leftEyeIdentifiers);

	auto landmarks = rcr_model.sdm_model.predict(init, cv::Mat(), hog);
	rcr::drawLandmarks(image, landmarks);
	cv::imwrite(outputfile.string(), image);

	return EXIT_SUCCESS;
}
