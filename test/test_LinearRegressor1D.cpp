#include "gtest/gtest.h"

#include "superviseddescent/regressors.hpp"

#include "opencv2/core/core.hpp"

using namespace superviseddescent;

TEST(LinearRegressor, OneDimOneExampleNoBiasLearning0) {
	using cv::Mat;
	Mat data = Mat::ones(1, 1, CV_32FC1);
	Mat labels = Mat::ones(1, 1, CV_32FC1);
	LinearRegressor<> lr;
	bool isInvertible = lr.learn(data, labels);
	EXPECT_EQ(true, isInvertible);
	ASSERT_FLOAT_EQ(1.0f, lr.x.at<float>(0)) << "Expected the learned x to be 1.0f";
}

TEST(LinearRegressor, OneDimOneExampleNoBiasLearning1) {
	using cv::Mat;
	Mat data = Mat::ones(1, 1, CV_32FC1);
	Mat labels = 0.5f * Mat::ones(1, 1, CV_32FC1);
	LinearRegressor<> lr;
	bool isInvertible = lr.learn(data, labels);
	EXPECT_EQ(true, isInvertible);
	ASSERT_FLOAT_EQ(0.5f, lr.x.at<float>(0)) << "Expected the learned x to be 0.5f";
}

/*
TEST(LinearRegressor, OneDimOneExampleNoBiasLearningNotInvertible) {
	using cv::Mat;
	Mat data = Mat::zeros(1, 1, CV_32FC1);
	Mat labels = Mat::ones(1, 1, CV_32FC1);
	LinearRegressor<> lr;
	bool isInvertible = lr.learn(data, labels);
	ASSERT_EQ(false, isInvertible);
}
*/

TEST(LinearRegressor, OneDimOneExampleNoBiasPrediction) {
	using cv::Mat;
	// Note/Todo: Load from filesystem, or from memory-bytes?
	Mat data = Mat::ones(1, 1, CV_32FC1);
	Mat labels = Mat::ones(1, 1, CV_32FC1);
	LinearRegressor<> lr;
	lr.learn(data, labels);

	// Test starts here:
	Mat test(1, 1, CV_32FC1);
	test.at<float>(0) = 0.0f;
	Mat prediction = lr.predict(test);
	EXPECT_FLOAT_EQ(0.0f, prediction.at<float>(0)) << "Expected the prediction to be 0.0f";

	test.at<float>(0) = 1.0f;
	prediction = lr.predict(test);
	EXPECT_FLOAT_EQ(1.0f, prediction.at<float>(0)) << "Expected the prediction to be 1.0f";

	test.at<float>(0) = 2.0f;
	prediction = lr.predict(test);
	ASSERT_FLOAT_EQ(2.0f, prediction.at<float>(0)) << "Expected the prediction to be 2.0f";
}

TEST(LinearRegressor, OneDimOneExampleNoBiasTestingNoResidual) {
	using cv::Mat;
	// Note/Todo: Load from filesystem, or from memory-bytes?
	Mat data = Mat::ones(1, 1, CV_32FC1);
	Mat labels = Mat::ones(1, 1, CV_32FC1);
	LinearRegressor<> lr;
	lr.learn(data, labels);

	// Test starts here:
	Mat test(3, 1, CV_32FC1);
	test.at<float>(0) = 0.0f;
	test.at<float>(1) = 1.0f;
	test.at<float>(2) = 2.0f;
	Mat groundtruth(3, 1, CV_32FC1);
	groundtruth.at<float>(0) = 0.0f;
	groundtruth.at<float>(1) = 1.0f;
	groundtruth.at<float>(2) = 2.0f;
	double residual = lr.test(test, groundtruth);
	ASSERT_DOUBLE_EQ(0.0, residual) << "Expected the residual to be 0.0";
}

TEST(LinearRegressor, OneDimOneExampleNoBiasTestingResidual) {
	using cv::Mat;
	// Note/Todo: Load from filesystem, or from memory-bytes?
	Mat data = Mat::ones(1, 1, CV_32FC1);
	Mat labels = Mat::ones(1, 1, CV_32FC1);
	LinearRegressor<> lr;
	lr.learn(data, labels);

	// Test starts here:
	Mat test(3, 1, CV_32FC1);
	test.at<float>(0) = 0.0f;
	test.at<float>(1) = 1.0f;
	test.at<float>(2) = 2.0f;
	Mat groundtruth(3, 1, CV_32FC1);
	groundtruth.at<float>(0) = -1.0f;
	groundtruth.at<float>(1) = 2.0f;
	groundtruth.at<float>(2) = 2.0f;
	double residual = lr.test(test, groundtruth);
	ASSERT_DOUBLE_EQ(0.47140452079103173, residual) << "Expected the residual to be 0.4714...";
}
