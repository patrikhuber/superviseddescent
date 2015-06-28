/*
 * superviseddescent: A C++11 implementation of the supervised descent
 *                    optimisation method
 * File: examples/pose_estimation.cpp
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

#include "opencv2/core/core.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif

#include <iostream>
#include <vector>
#include <random>

using namespace superviseddescent;
using cv::Mat;
using cv::Vec4f;
using std::vector;
using std::cout;
using std::endl;

float rad2deg(float radians)
{
	return radians * static_cast<float>(180 / CV_PI);
}

float deg2rad(float degrees)
{
	return degrees * static_cast<float>(CV_PI / 180);
}

float focalLengthToFovy(float focalLength, float height)
{
	return rad2deg(2.0f * std::atan2(height, 2.0f * focalLength));
}


/**
 * Creates a 4x4 rotation matrix with a rotation in \p angle radians
 * around the x axis (pitch angle). Following OpenGL and Qt's conventions.
 *
 * @param[in] angle The angle around the x axis in radians.
 */
cv::Mat createRotationMatrixX(float angle)
{
	cv::Mat rotX = (cv::Mat_<float>(4, 4) <<
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, std::cos(angle), -std::sin(angle), 0.0f,
		0.0f, std::sin(angle), std::cos(angle), 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);
	return rotX;
}

/**
 * Creates a 4x4 rotation matrix with a rotation in \p angle radians
 * around the y axis (yaw angle). Following OpenGL and Qt's conventions.
 *
 * @param[in] angle The angle around the y axis in radians.
 */
cv::Mat createRotationMatrixY(float angle)
{
	cv::Mat rotY = (cv::Mat_<float>(4, 4) <<
		std::cos(angle), 0.0f, std::sin(angle), 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		-std::sin(angle), 0.0f, std::cos(angle), 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);
	return rotY;
}

/**
 * Creates a 4x4 rotation matrix with a rotation in \p angle radians
 * around the z axis (roll angle). Following OpenGL and Qt's conventions.
 *
 * @param[in] angle The angle around the z axis in radians.
 */
cv::Mat createRotationMatrixZ(float angle)
{
	cv::Mat rotZ = (cv::Mat_<float>(4, 4) <<
		std::cos(angle), -std::sin(angle), 0.0f, 0.0f,
		std::sin(angle), std::cos(angle), 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);
	return rotZ;
}

/**
 * Creates a 4x4 scaling matrix that scales homogeneous vectors along x, y and z.
 *
 * @param[in] sx Scaling along the x direction.
 * @param[in] sy Scaling along the y direction.
 * @param[in] sz Scaling along the z direction.
 */
cv::Mat createScalingMatrix(float sx, float sy, float sz)
{
	cv::Mat scaling = (cv::Mat_<float>(4, 4) <<
		sx, 0.0f, 0.0f, 0.0f,
		0.0f, sy, 0.0f, 0.0f,
		0.0f, 0.0f, sz, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);
	return scaling;
}

/**
 * Creates a 4x4 translation matrix that scales homogeneous vectors along x, y and z direction.
 *
 * @param[in] tx Translation along the x direction.
 * @param[in] ty Translation along the y direction.
 * @param[in] tz Translation along the z direction.
 */
cv::Mat createTranslationMatrix(float tx, float ty, float tz)
{
	cv::Mat translation = (cv::Mat_<float>(4, 4) <<
		1.0f, 0.0f, 0.0f, tx,
		0.0f, 1.0f, 0.0f, ty,
		0.0f, 0.0f, 1.0f, tz,
		0.0f, 0.0f, 0.0f, 1.0f);
	return translation;
}

/**
 * Creates a 4x4 perspective projection matrix, following OpenGL & Qt conventions.
 *
 * @param[in] verticalAngle Vertical angle of the FOV, in degrees.
 * @param[in] aspectRatio Aspect ratio of the camera.
 * @param[in] n Near plane distance.
 * @param[in] f Far plane distance.
 */
cv::Mat createPerspectiveProjectionMatrix(float verticalAngle, float aspectRatio, float n, float f)
{
	float radians = (verticalAngle / 2.0f) * static_cast<float>(CV_PI) / 180.0f;
	float sine = std::sin(radians);
	// if sinr == 0.0f, return, something wrong
	float cotan = std::cos(radians) / sine;
	cv::Mat perspective = (cv::Mat_<float>(4, 4) <<
		cotan / aspectRatio, 0.0f, 0.0f, 0.0f,
		0.0f, cotan, 0.0f, 0.0f,
		0.0f, 0.0f, -(n + f) / (f - n), (-2.0f * n * f) / (f - n),
		0.0f, 0.0f, -1.0f, 0.0f);
	return perspective;
}

/**
 * Projects a vertex in homogeneous coordinates to screen coordinates.
 *
 * @param[in] vertex A vertex in homogeneous coordinates.
 * @param[in] model_view_projection A 4x4 model-view-projection matrix.
 * @param[in] screen_width Screen / window width.
 * @param[in] screen_height Screen / window height.
 */
cv::Vec3f projectVertex(cv::Vec4f vertex, cv::Mat model_view_projection, int screen_width, int screen_height)
{
	cv::Mat clipSpace = model_view_projection * cv::Mat(vertex);
	cv::Vec4f clipSpaceV(clipSpace);
	clipSpaceV = clipSpaceV / clipSpaceV[3]; // divide by w
	// Viewport transform:
	float x_ss = (clipSpaceV[0] + 1.0f) * (screen_width / 2.0f);
	float y_ss = screen_height - (clipSpaceV[1] + 1.0f) * (screen_height / 2.0f); // flip the y axis, image origins are different
	cv::Vec2f screenSpace(x_ss, y_ss);
	return cv::Vec3f(screenSpace[0], screenSpace[1], clipSpaceV[2]);
}

/**
 * Function object that projects a 3D model to 2D, given a set of
 * parameters [R_x, R_y, R_z, t_x, t_y, t_z]. 
 *
 * It is used in this example for the 6 DOF pose estimation.
 * A perspective camera with focal length 1800 and a screen of
 * 1000 x 1000, with the camera at [500, 500], is used in this example.
 *
 * The 2D points are normalised after projection by subtracting the camera
 * origin and dividing by the focal length.
 */
class ModelProjection
{
public:
	/**
	 * Constructs a new projection object with the given model.
	 *
	 * @param[in] model 3D model points in a 4 x n matrix, where n is the number of points, and the points are in homgeneous coordinates.
	 */
	ModelProjection(cv::Mat model) : model(model)
	{
	};

	/**
	 * Uses the current parameters ([R, t], in SDM terminology the \c x) to
	 * project the points from the 3D model to 2D space. The 2D points are
	 * the new \c y.
	 *
	 * The current parameter estimate is given as a row vector
	 * [R_x, R_y, R_z, t_x, t_y, t_z].
	 *
	 * @param[in] parameters The current estimate of the 6 DOF.
	 * @param[in] regressorLevel Not used in this example.
	 * @param[in] trainingIndex Not used in this example.
	 * @return Returns normalised projected 2D landmark coordinates.
	 */
	cv::Mat operator()(cv::Mat parameters, size_t regressorLevel, int trainingIndex = 0)
	{
		assert((parameters.cols == 1 && parameters.rows == 6) || (parameters.cols == 6 && parameters.rows == 1));
		using cv::Mat;
		// Project the 3D model points using the current parameters:
		float focalLength = 1800.0f;
		Mat rotPitchX = createRotationMatrixX(deg2rad(parameters.at<float>(0)));
		Mat rotYawY = createRotationMatrixY(deg2rad(parameters.at<float>(1)));
		Mat rotRollZ = createRotationMatrixZ(deg2rad(parameters.at<float>(2)));
		Mat translation = createTranslationMatrix(parameters.at<float>(3), parameters.at<float>(4), parameters.at<float>(5));
		Mat modelMatrix = translation * rotYawY * rotPitchX * rotRollZ;
		const float aspect = static_cast<float>(1000) / static_cast<float>(1000);
		float fovY = focalLengthToFovy(focalLength, 1000);
		Mat projectionMatrix = createPerspectiveProjectionMatrix(fovY, aspect, 1.0f, 5000.0f);

		int numLandmarks = model.cols;
		Mat new2dProjections(1, numLandmarks * 2, CV_32FC1);
		for (int lm = 0; lm < numLandmarks; ++lm) {
			cv::Vec3f vtx2d = projectVertex(cv::Vec4f(model.col(lm)), projectionMatrix * modelMatrix, 1000, 1000);
			// 'Normalise' the image coordinates of the projection by subtracting the origin (center of the image) and dividing by f:
			vtx2d = (vtx2d - cv::Vec3f(500.0f, 500.0f)) / focalLength;
			new2dProjections.at<float>(lm) = vtx2d[0]; // the x coord
			new2dProjections.at<float>(lm + numLandmarks) = vtx2d[1]; // y coord
		}
		return new2dProjections;
	};
private:
	cv::Mat model;
};

/**
 * This app demonstrates learning of the descent direction from data for
 * a simple 6 degree of freedom face pose estimation.
 *
 * It uses a simple 10-point 3D face model, generates random poses and
 * uses the generated pose parameters (\c x) and their respective 2D
 * projections (\c y) to learn how to optimise for the parameters given
 * input landmarks \c y.
 *
 * This is an example of the library when a known template \c y is available
 * for training and testing.
 */
int main(int argc, char *argv[])
{
	Mat facemodel; // The point numbers are from the iBug landmarking scheme
	facemodel.push_back(Vec4f(-0.287526f, -2.0203f, 3.33725f, 1.0f)); // nose tip, 31
	facemodel.push_back(Vec4f(-0.11479f, -17.2056f, -13.5569f, 1.0f)); // nose-lip junction, 34
	facemodel.push_back(Vec4f(-46.1668f, 34.7219f, -35.938f, 1.0f)); // right eye outer corner, 37
	facemodel.push_back(Vec4f(-18.926f, 31.5432f, -29.9641f, 1.0f)); // right eye inner corner, 40
	facemodel.push_back(Vec4f(19.2574f, 31.5767f, -30.229f, 1.0f)); // left eye inner corner, 43
	facemodel.push_back(Vec4f(46.1914f, 34.452f, -36.1317f, 1.0f)); // left eye outer corner, 46
	facemodel.push_back(Vec4f(-23.7552f, -35.7461f, -28.2573f, 1.0f)); // mouth right corner, 49
	facemodel.push_back(Vec4f(-0.0753515f, -28.3064f, -12.8984f, 1.0f)); // upper lip center top, 52
	facemodel.push_back(Vec4f(23.7138f, -35.7886f, -28.5949f, 1.0f)); // mouth left corner, 55
	facemodel.push_back(Vec4f(0.125511f, -44.7427f, -17.1411f, 1.0f)); // lower lip center bottom, 58
	facemodel = facemodel.reshape(1, 10).t(); // reshape to 1 channel, 10 rows, then transpose

	// Random generator for a random angle in [-30, 30]:
	std::random_device rd;
	std::mt19937 engine(rd());
	std::uniform_real_distribution<float> angle_distribution(-30, 30);
	auto random_angle = [&angle_distribution, &engine]() { return angle_distribution(engine); };
	
	// For the sake of a brief example, we only sample the angles and keep the x and y translation constant:
	float tx = 0.0f;
	float ty = 0.0f;
	float tz = -2000.0f;

	vector<LinearRegressor<>> regressors;
	regressors.emplace_back(LinearRegressor<>(Regulariser(Regulariser::RegularisationType::MatrixNorm, 2.0f, true)));
	regressors.emplace_back(LinearRegressor<>(Regulariser(Regulariser::RegularisationType::MatrixNorm, 2.0f, true)));
	regressors.emplace_back(LinearRegressor<>(Regulariser(Regulariser::RegularisationType::MatrixNorm, 2.0f, true)));

	SupervisedDescentOptimiser<LinearRegressor<>> supervised_descent_model(regressors);

	ModelProjection projection(facemodel);

	// Generate 500 random parameter samples, consisting of
	// the 6 DOF. Stored as [r_x, r_y, r_z, t_x, t_y, t_z]:
	int num_samples = 500;
	Mat x_tr(num_samples, 6, CV_32FC1);
	for (int row = 0; row < num_samples; ++row) {
		x_tr.at<float>(row, 0) = random_angle();
		x_tr.at<float>(row, 1) = random_angle();
		x_tr.at<float>(row, 2) = random_angle();
		x_tr.at<float>(row, 3) = tx;
		x_tr.at<float>(row, 4) = ty;
		x_tr.at<float>(row, 5) = tz;
	}

	// Calculate and store the corresponding 2D landmark projections:
	// Note: In a real application, we would add some noise to the data.
	auto num_landmarks = facemodel.cols;
	Mat y_tr(num_samples, 2 * num_landmarks, CV_32FC1);
	for (int row = 0; row < num_samples; ++row) {
		auto landmarks = projection(x_tr.row(row), 0);
		landmarks.copyTo(y_tr.row(row));
	}

	Mat x0 = Mat::zeros(num_samples, 6, CV_32FC1); // fixed initialisation of the parameters, all zero, except t_z
	x0.col(5) = -2000.0f;

	auto print_residual = [&x_tr](const cv::Mat& current_predictions) {
		cout << cv::norm(current_predictions, x_tr, cv::NORM_L2) / cv::norm(x_tr, cv::NORM_L2) << endl;
	};
	// Train the model. We'll also specify an optional callback function:
	cout << "Training the model, printing the residual after each learned regressor: " << endl;
	supervised_descent_model.train(x_tr, x0, y_tr, projection, print_residual);

	// Test: Omitted for brevity, as we didn't generate test data
	//Mat predictions = supervisedDescentModel.test(x0, y_tr, projection, printResidual);

	// Prediction on new landmarks, [x_0, ..., x_n, y_0, ..., y_n]:
	Mat landmarks = (cv::Mat_<float>(1, 20) << 498.0f, 504.0f, 479.0f, 498.0f, 529.0f, 553.0f, 489.0f, 503.0f, 527.0f, 503.0f, 502.0f, 513.0f, 457.0f, 465.0f, 471.0f, 471.0f, 522.0f, 522.0f, 530.0f, 536.0f);
	// Normalise the coordinates w.r.t. the image origin and focal length (we do the same during training):
	landmarks = (landmarks - 500.0f) / 1800.0f;
	
	Mat initial_params = Mat::zeros(1, 6, CV_32FC1);
	initial_params.at<float>(5) = -2000.0f; // [0, 0, 0, 0, 0, -2000]
	
	Mat predicted_params = supervised_descent_model.predict(initial_params, landmarks, projection);
	cout << "Groundtruth pose: pitch = 11.0, yaw = -25.0, roll = -10.0" << endl;
	cout << "Predicted pose: pitch = " << predicted_params.at<float>(0) << ", yaw = " << predicted_params.at<float>(1) << ", roll = " << predicted_params.at<float>(2) << endl;
	
	return EXIT_SUCCESS;
}
