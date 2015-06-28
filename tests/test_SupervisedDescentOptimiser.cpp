#include "gtest/gtest.h"

#include "superviseddescent/superviseddescent.hpp"
#include "superviseddescent/regressors.hpp"

#include "opencv2/core/core.hpp"
/*#ifdef WIN32
#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif*/
#include "boost/math/special_functions/erf.hpp"

#include <vector>

using cv::Mat;
using std::vector;
using namespace superviseddescent;

template<typename ForwardIterator, typename T>
void strided_iota(ForwardIterator first, ForwardIterator last, T value, T stride)
{
	while (first != last) {
		*first++ = value;
		value += stride;
	}
}

double normalisedLeastSquaresResidual(const Mat& prediction, const Mat& groundtruth)
{
	return cv::norm(prediction, groundtruth, cv::NORM_L2) / cv::norm(groundtruth, cv::NORM_L2);
}

TEST(SupervisedDescentOptimiser, SinConvergence) {
	// sin(x):
	auto h = [](Mat value, size_t, int) { return std::sin(value.at<float>(0)); };
	auto h_inv = [](float value) {
		if (value >= 1.0f) // our upper border of y is 1.0f, but it can be a bit larger due to floating point representation. asin then returns NaN.
			return std::asin(1.0f);
		else
			return std::asin(value);
		};

	float startInterval = -1.0f; float stepSize = 0.2f; int numValues = 11; Mat y_tr(numValues, 1, CV_32FC1); // sin: [-1:0.2:1]
	{
		vector<float> values(numValues);
		strided_iota(std::begin(values), std::next(std::begin(values), numValues), startInterval, stepSize);
		y_tr = Mat(values, true);
	}
	Mat x_tr(numValues, 1, CV_32FC1); // Will be the inverse of y_tr
	{
		vector<float> values(numValues);
		std::transform(y_tr.begin<float>(), y_tr.end<float>(), begin(values), h_inv);
		x_tr = Mat(values, true);
	}

	Mat x0 = 0.5f * Mat::ones(numValues, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	SupervisedDescentOptimiser<LinearRegressor<>> sdo({ LinearRegressor<>() });
	
	// Test the callback mechanism as well: (better move to a separate unit test?)
	auto checkResidual = [&](const Mat& currentX) {
		double residual = cv::norm(currentX, x_tr, cv::NORM_L2) / cv::norm(x_tr, cv::NORM_L2);
		EXPECT_DOUBLE_EQ(0.21369851877468238, residual);
	};

	sdo.train(x_tr, x0, y_tr, h, checkResidual);

	// Make sure the training converges, i.e. the residual is correct on the training data:
	Mat predictions = sdo.test(x0, y_tr, h);
	double trainingResidual = normalisedLeastSquaresResidual(predictions, x_tr);
	EXPECT_DOUBLE_EQ(0.21369851877468238, trainingResidual);

	// Test the trained model:
	// Test data with finer resolution:
	float startIntervalTest = -1.0f; float stepSizeTest = 0.05f; int numValuesTest = 41; Mat y_ts(numValuesTest, 1, CV_32FC1); // sin: [-1:0.05:1]
	{
		vector<float> values(numValuesTest);
		strided_iota(std::begin(values), std::next(std::begin(values), numValuesTest), startIntervalTest, stepSizeTest);
		y_ts = Mat(values, true);
	}
	Mat x_ts_gt(numValuesTest, 1, CV_32FC1); // Will be the inverse of y_ts
	{
		vector<float> values(numValuesTest);
		std::transform(y_ts.begin<float>(), y_ts.end<float>(), begin(values), h_inv);
		x_ts_gt = Mat(values, true);
	}
	Mat x0_ts = 0.5f * Mat::ones(numValuesTest, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.
	
	predictions = sdo.test(x0_ts, y_ts, h);
	double testResidual = normalisedLeastSquaresResidual(predictions, x_ts_gt);
	ASSERT_NEAR(0.1800101229, testResidual, 0.0000000003);
}

TEST(SupervisedDescentOptimiser, SinConvergenceCascade) {
	// sin(x):
	auto h = [](Mat value, size_t, int) { return std::sin(value.at<float>(0)); };
	auto h_inv = [](float value) {
		if (value >= 1.0f) // our upper border of y is 1.0f, but it can be a bit larger due to floating point representation. asin then returns NaN.
			return std::asin(1.0f);
		else
			return std::asin(value);
	};

	float startInterval = -1.0f; float stepSize = 0.2f; int numValues = 11; Mat y_tr(numValues, 1, CV_32FC1); // sin: [-1:0.2:1]
	{
		vector<float> values(numValues);
		strided_iota(std::begin(values), std::next(std::begin(values), numValues), startInterval, stepSize);
		y_tr = Mat(values, true);
	}
	Mat x_tr(numValues, 1, CV_32FC1); // Will be the inverse of y_tr
	{
		vector<float> values(numValues);
		std::transform(y_tr.begin<float>(), y_tr.end<float>(), begin(values), h_inv);
		x_tr = Mat(values, true);
	}

	Mat x0 = 0.5f * Mat::ones(numValues, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	vector<LinearRegressor<>> regressors(10);
	SupervisedDescentOptimiser<LinearRegressor<>> sdo(regressors);
	sdo.train(x_tr, x0, y_tr, h);

	// Make sure the training converges, i.e. the residual is correct on the training data:
	Mat predictions = sdo.test(x0, y_tr, h);
	double trainingResidual = normalisedLeastSquaresResidual(predictions, x_tr);
	EXPECT_NEAR(0.040279395, trainingResidual, 0.00000008);
		
	// Test the trained model:
	// Test data with finer resolution:
	float startIntervalTest = -1.0f; float stepSizeTest = 0.05f; int numValuesTest = 41; Mat y_ts(numValuesTest, 1, CV_32FC1); // sin: [-1:0.05:1]
	{
		vector<float> values(numValuesTest);
		strided_iota(std::begin(values), std::next(std::begin(values), numValuesTest), startIntervalTest, stepSizeTest);
		y_ts = Mat(values, true);
	}
	Mat x_ts_gt(numValuesTest, 1, CV_32FC1); // Will be the inverse of y_ts
	{
		vector<float> values(numValuesTest);
		std::transform(y_ts.begin<float>(), y_ts.end<float>(), begin(values), h_inv);
		x_ts_gt = Mat(values, true);
	}
	Mat x0_ts = 0.5f * Mat::ones(numValuesTest, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	predictions = sdo.test(x0_ts, y_ts, h);
	double testResidual = normalisedLeastSquaresResidual(predictions, x_ts_gt);
	ASSERT_NEAR(0.026156775, testResidual, 0.00000005);
}

TEST(SupervisedDescentOptimiser, XCubeConvergence) {
	// x^3:
	auto h = [](Mat value, size_t, int) { return static_cast<float>(std::pow(value.at<float>(0), 3)); };
	auto h_inv = [](float value) { return std::cbrt(value); }; // cubic root

	float startInterval = -27.0f; float stepSize = 3.0f; int numValues = 19; Mat y_tr(numValues, 1, CV_32FC1); // cube: [-27:3:27]
	{
		vector<float> values(numValues);
		strided_iota(std::begin(values), std::next(std::begin(values), numValues), startInterval, stepSize);
		y_tr = Mat(values, true);
	}
	Mat x_tr(numValues, 1, CV_32FC1); // Will be the inverse of y_tr
	{
		vector<float> values(numValues);
		std::transform(y_tr.begin<float>(), y_tr.end<float>(), begin(values), h_inv);
		x_tr = Mat(values, true);
	}

	Mat x0 = 0.5f * Mat::ones(numValues, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	SupervisedDescentOptimiser<LinearRegressor<>> sdo({ LinearRegressor<>() });
	sdo.train(x_tr, x0, y_tr, h);

	// Make sure the training converges, i.e. the residual is correct on the training data:
	Mat predictions = sdo.test(x0, y_tr, h);
	double trainingResidual = normalisedLeastSquaresResidual(predictions, x_tr);
	EXPECT_NEAR(0.34416553, trainingResidual, 0.00000002);

	// Test the trained model:
	// Test data with finer resolution:
	float startIntervalTest = -27.0f; float stepSizeTest = 0.5f; int numValuesTest = 109; Mat y_ts(numValuesTest, 1, CV_32FC1); // cube: [-27:0.5:27]
	{
		vector<float> values(numValuesTest);
		strided_iota(std::begin(values), std::next(std::begin(values), numValuesTest), startIntervalTest, stepSizeTest);
		y_ts = Mat(values, true);
	}
	Mat x_ts_gt(numValuesTest, 1, CV_32FC1); // Will be the inverse of y_ts
	{
		vector<float> values(numValuesTest);
		std::transform(y_ts.begin<float>(), y_ts.end<float>(), begin(values), h_inv);
		x_ts_gt = Mat(values, true);
	}
	Mat x0_ts = 0.5f * Mat::ones(numValuesTest, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	predictions = sdo.test(x0_ts, y_ts, h);
	double testResidual = normalisedLeastSquaresResidual(predictions, x_ts_gt);
	ASSERT_NEAR(0.353428615, testResidual, 0.00002);
}

TEST(SupervisedDescentOptimiser, XCubeConvergenceCascade) {
	// x^3:
	auto h = [](Mat value, size_t, int) { return static_cast<float>(std::pow(value.at<float>(0), 3)); };
	auto h_inv = [](float value) { return std::cbrt(value); }; // cubic root

	float startInterval = -27.0f; float stepSize = 3.0f; int numValues = 19; Mat y_tr(numValues, 1, CV_32FC1); // cube: [-27:3:27]
	{
		vector<float> values(numValues);
		strided_iota(std::begin(values), std::next(std::begin(values), numValues), startInterval, stepSize);
		y_tr = Mat(values, true);
	}
	Mat x_tr(numValues, 1, CV_32FC1); // Will be the inverse of y_tr
	{
		vector<float> values(numValues);
		std::transform(y_tr.begin<float>(), y_tr.end<float>(), begin(values), h_inv);
		x_tr = Mat(values, true);
	}

	Mat x0 = 0.5f * Mat::ones(numValues, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	vector<LinearRegressor<>> regressors(10);
	SupervisedDescentOptimiser<LinearRegressor<>> sdo(regressors);
	sdo.train(x_tr, x0, y_tr, h);

	// Make sure the training converges, i.e. the residual is correct on the training data:
	Mat predictions = sdo.test(x0, y_tr, h);
	double trainingResidual = normalisedLeastSquaresResidual(predictions, x_tr);
	EXPECT_NEAR(0.04312725, trainingResidual, 0.00000002);

	// Test the trained model:
	// Test data with finer resolution:
	float startIntervalTest = -27.0f; float stepSizeTest = 0.5f; int numValuesTest = 109; Mat y_ts(numValuesTest, 1, CV_32FC1); // cube: [-27:0.5:27]
	{
		vector<float> values(numValuesTest);
		strided_iota(std::begin(values), std::next(std::begin(values), numValuesTest), startIntervalTest, stepSizeTest);
		y_ts = Mat(values, true);
	}
	Mat x_ts_gt(numValuesTest, 1, CV_32FC1); // Will be the inverse of y_ts
	{
		vector<float> values(numValuesTest);
		std::transform(y_ts.begin<float>(), y_ts.end<float>(), begin(values), h_inv);
		x_ts_gt = Mat(values, true);
	}
	Mat x0_ts = 0.5f * Mat::ones(numValuesTest, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	predictions = sdo.test(x0_ts, y_ts, h);
	double testResidual = normalisedLeastSquaresResidual(predictions, x_ts_gt);
	ASSERT_NEAR(0.05889855, testResidual, 0.00000002);
}

TEST(SupervisedDescentOptimiser, ErfConvergence) {
	// erf(x):
	auto h = [](Mat value, size_t, int) { return std::erf(value.at<float>(0)); };
	auto h_inv = [](float value) { return boost::math::erf_inv(value); };

	float startInterval = -0.99f; float stepSize = 0.11f; int numValues = 19; Mat y_tr(numValues, 1, CV_32FC1); // erf: [-0.99:0.11:0.99]
	{
		vector<float> values(numValues);
		strided_iota(std::begin(values), std::next(std::begin(values), numValues), startInterval, stepSize);
		y_tr = Mat(values, true);
	}
	Mat x_tr(numValues, 1, CV_32FC1); // Will be the inverse of y_tr
	{
		vector<float> values(numValues);
		std::transform(y_tr.begin<float>(), y_tr.end<float>(), begin(values), h_inv);
		x_tr = Mat(values, true);
	}

	Mat x0 = 0.5f * Mat::ones(numValues, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	SupervisedDescentOptimiser<LinearRegressor<>> sdo({ LinearRegressor<>() });
	sdo.train(x_tr, x0, y_tr, h);

	// Make sure the training converges, i.e. the residual is correct on the training data:
	Mat predictions = sdo.test(x0, y_tr, h);
	double trainingResidual = normalisedLeastSquaresResidual(predictions, x_tr);
	EXPECT_NEAR(0.30944183, trainingResidual, 0.00000005);

	// Test the trained model:
	// Test data with finer resolution:
	float startIntervalTest = -0.99f; float stepSizeTest = 0.03f; int numValuesTest = 67; Mat y_ts(numValuesTest, 1, CV_32FC1); // erf: [-0.99:0.03:0.99]
	{
		vector<float> values(numValuesTest);
		strided_iota(std::begin(values), std::next(std::begin(values), numValuesTest), startIntervalTest, stepSizeTest);
		y_ts = Mat(values, true);
	}
	Mat x_ts_gt(numValuesTest, 1, CV_32FC1); // Will be the inverse of y_ts
	{
		vector<float> values(numValuesTest);
		std::transform(y_ts.begin<float>(), y_ts.end<float>(), begin(values), h_inv);
		x_ts_gt = Mat(values, true);
	}
	Mat x0_ts = 0.5f * Mat::ones(numValuesTest, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	predictions = sdo.test(x0_ts, y_ts, h);
	double testResidual = normalisedLeastSquaresResidual(predictions, x_ts_gt);
	ASSERT_NEAR(0.25736006, testResidual, 0.0000002);
}

TEST(SupervisedDescentOptimiser, ErfConvergenceCascade) {
	// erf(x):
	auto h = [](Mat value, size_t, int) { return std::erf(value.at<float>(0)); };
	auto h_inv = [](float value) { return boost::math::erf_inv(value); };

	float startInterval = -0.99f; float stepSize = 0.11f; int numValues = 19; Mat y_tr(numValues, 1, CV_32FC1); // erf: [-0.99:0.11:0.99]
	{
		vector<float> values(numValues);
		strided_iota(std::begin(values), std::next(std::begin(values), numValues), startInterval, stepSize);
		y_tr = Mat(values, true);
	}
	Mat x_tr(numValues, 1, CV_32FC1); // Will be the inverse of y_tr
	{
		vector<float> values(numValues);
		std::transform(y_tr.begin<float>(), y_tr.end<float>(), begin(values), h_inv);
		x_tr = Mat(values, true);
	}

	Mat x0 = 0.5f * Mat::ones(numValues, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	vector<LinearRegressor<>> regressors(10);
	SupervisedDescentOptimiser<LinearRegressor<>> sdo(regressors);
	sdo.train(x_tr, x0, y_tr, h);

	// Make sure the training converges, i.e. the residual is correct on the training data:
	Mat predictions = sdo.test(x0, y_tr, h);
	double trainingResidual = normalisedLeastSquaresResidual(predictions, x_tr);
	EXPECT_NEAR(0.06951067, trainingResidual, 0.00000005);

	// Test the trained model:
	// Test data with finer resolution:
	float startIntervalTest = -0.99f; float stepSizeTest = 0.03f; int numValuesTest = 67; Mat y_ts(numValuesTest, 1, CV_32FC1); // erf: [-0.99:0.03:0.99]
	{
		vector<float> values(numValuesTest);
		strided_iota(std::begin(values), std::next(std::begin(values), numValuesTest), startIntervalTest, stepSizeTest);
		y_ts = Mat(values, true);
	}
	Mat x_ts_gt(numValuesTest, 1, CV_32FC1); // Will be the inverse of y_ts
	{
		vector<float> values(numValuesTest);
		std::transform(y_ts.begin<float>(), y_ts.end<float>(), begin(values), h_inv);
		x_ts_gt = Mat(values, true);
	}
	Mat x0_ts = 0.5f * Mat::ones(numValuesTest, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	predictions = sdo.test(x0_ts, y_ts, h);
	double testResidual = normalisedLeastSquaresResidual(predictions, x_ts_gt);
	ASSERT_NEAR(0.04632717, testResidual, 0.00000005);
}

TEST(SupervisedDescentOptimiser, ExpConvergence) {
	// exp(x):
	auto h = [](Mat value, size_t, int) { return std::exp(value.at<float>(0)); };
	auto h_inv = [](float value) { return std::log(value); };

	float startInterval = 1.0f; float stepSize = 3.0f; int numValues = 10; Mat y_tr(numValues, 1, CV_32FC1); // exp: [1:3:28]
	{
		vector<float> values(numValues);
		strided_iota(std::begin(values), std::next(std::begin(values), numValues), startInterval, stepSize);
		y_tr = Mat(values, true);
	}
	Mat x_tr(numValues, 1, CV_32FC1); // Will be the inverse of y_tr
	{
		vector<float> values(numValues);
		std::transform(y_tr.begin<float>(), y_tr.end<float>(), begin(values), h_inv);
		x_tr = Mat(values, true);
	}

	Mat x0 = 0.5f * Mat::ones(numValues, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	SupervisedDescentOptimiser<LinearRegressor<>> sdo({ LinearRegressor<>() });
	sdo.train(x_tr, x0, y_tr, h);

	// Make sure the training converges, i.e. the residual is correct on the training data:
	Mat predictions = sdo.test(x0, y_tr, h);
	double trainingResidual = normalisedLeastSquaresResidual(predictions, x_tr);
	EXPECT_NEAR(0.19952251597692217, trainingResidual, 0.00000002);

	// Test the trained model:
	// Test data with finer resolution:
	float startIntervalTest = 1.0f; float stepSizeTest = 0.5f; int numValuesTest = 55; Mat y_ts(numValuesTest, 1, CV_32FC1); // exp: [1:0.5:28]
	{
		vector<float> values(numValuesTest);
		strided_iota(std::begin(values), std::next(std::begin(values), numValuesTest), startIntervalTest, stepSizeTest);
		y_ts = Mat(values, true);
	}
	Mat x_ts_gt(numValuesTest, 1, CV_32FC1); // Will be the inverse of y_ts
	{
		vector<float> values(numValuesTest);
		std::transform(y_ts.begin<float>(), y_ts.end<float>(), begin(values), h_inv);
		x_ts_gt = Mat(values, true);
	}
	Mat x0_ts = 0.5f * Mat::ones(numValuesTest, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	predictions = sdo.test(x0_ts, y_ts, h);
	double testResidual = normalisedLeastSquaresResidual(predictions, x_ts_gt);
	ASSERT_NEAR(0.1924569501, testResidual, 0.000000006);
}

TEST(SupervisedDescentOptimiser, ExpConvergenceCascade) {
	// exp(x):
	auto h = [](Mat value, size_t, int) { return std::exp(value.at<float>(0)); };
	auto h_inv = [](float value) { return std::log(value); };

	float startInterval = 1.0f; float stepSize = 3.0f; int numValues = 10; Mat y_tr(numValues, 1, CV_32FC1); // exp: [1:3:28]
	{
		vector<float> values(numValues);
		strided_iota(std::begin(values), std::next(std::begin(values), numValues), startInterval, stepSize);
		y_tr = Mat(values, true);
	}
	Mat x_tr(numValues, 1, CV_32FC1); // Will be the inverse of y_tr
	{
		vector<float> values(numValues);
		std::transform(y_tr.begin<float>(), y_tr.end<float>(), begin(values), h_inv);
		x_tr = Mat(values, true);
	}

	Mat x0 = 0.5f * Mat::ones(numValues, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	vector<LinearRegressor<>> regressors(10);
	SupervisedDescentOptimiser<LinearRegressor<>> sdo(regressors);
	sdo.train(x_tr, x0, y_tr, h);

	// Make sure the training converges, i.e. the residual is correct on the training data:
	Mat predictions = sdo.test(x0, y_tr, h);
	double trainingResidual = normalisedLeastSquaresResidual(predictions, x_tr);
	EXPECT_NEAR(0.02510868, trainingResidual, 0.00000005);
	
	// Test the trained model:
	// Test data with finer resolution:
	float startIntervalTest = 1.0f; float stepSizeTest = 0.5f; int numValuesTest = 55; Mat y_ts(numValuesTest, 1, CV_32FC1); // exp: [1:0.5:28]
	{
		vector<float> values(numValuesTest);
		strided_iota(std::begin(values), std::next(std::begin(values), numValuesTest), startIntervalTest, stepSizeTest);
		y_ts = Mat(values, true);
	}
	Mat x_ts_gt(numValuesTest, 1, CV_32FC1); // Will be the inverse of y_ts
	{
		vector<float> values(numValuesTest);
		std::transform(y_ts.begin<float>(), y_ts.end<float>(), begin(values), h_inv);
		x_ts_gt = Mat(values, true);
	}
	Mat x0_ts = 0.5f * Mat::ones(numValuesTest, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	predictions = sdo.test(x0_ts, y_ts, h);
	double testResidual = normalisedLeastSquaresResidual(predictions, x_ts_gt);
	ASSERT_NEAR(0.01253494, testResidual, 0.00000004);
}

TEST(SupervisedDescentOptimiser, SinErfConvergenceCascadeMultiY) {
	// sin(x):
	auto h_sin = [](float value) { return std::sin(value); };
	auto h_sin_inv = [](float value) {
		if (value >= 1.0f) // our upper border of y is 1.0f, but it can be a bit larger due to floating point representation. asin then returns NaN.
			return std::asin(1.0f);
		else
			return std::asin(value);
	};
	// erf(x):
	auto h_erf = [](float value) { return std::erf(value); };
	auto h_erf_inv = [](float value) { return boost::math::erf_inv(value); };

	auto h = [&](Mat value, size_t, int) {
		Mat result(1, 2, CV_32FC1);
		result.at<float>(0) = h_sin(value.at<float>(0));
		result.at<float>(1) = h_erf(value.at<float>(1));
		return result;
	};
	/*
	auto h_inv = [&](Mat value) {
		Mat result(1, 2, CV_32FC1);
		result.at<float>(0) = h_sin_inv(value.at<float>(0));
		result.at<float>(1) = h_erf_inv(value.at<float>(1));
		return result;
	};*/

	float startInterval = -0.99f; float stepSize = 0.11f; int numValues = 19; Mat y_tr(numValues, 2, CV_32FC1); // should work for sin and erf: [-0.99:0.11:0.99]
	{
		vector<float> values(numValues);
		strided_iota(std::begin(values), std::next(std::begin(values), numValues), startInterval, stepSize);
		for (auto r = 0; r < y_tr.rows; ++r) {
			y_tr.at<float>(r, 0) = values[r];
			y_tr.at<float>(r, 1) = values[r];
		}

	}
	Mat x_tr(numValues, 2, CV_32FC1); // Will be the inverse of y_tr
	{
		for (auto r = 0; r < x_tr.rows; ++r) {
			x_tr.at<float>(r, 0) = h_sin_inv(y_tr.at<float>(r, 0));
			x_tr.at<float>(r, 1) = h_erf_inv(y_tr.at<float>(r, 1));
		}
	}

	Mat x0 = 0.5f * Mat::ones(numValues, 2, CV_32FC1); // fixed initialization x0 = c = 0.5.

	vector<LinearRegressor<>> regressors(10);
	SupervisedDescentOptimiser<LinearRegressor<>> sdo(regressors);

	sdo.train(x_tr, x0, y_tr, h);
	Mat predictions = sdo.test(x0, y_tr, h);
	double trainingResidual = cv::norm(predictions, x_tr, cv::NORM_L2) / cv::norm(x_tr, cv::NORM_L2);
	EXPECT_NEAR(0.0002677, trainingResidual, 0.0000004);
	
	// Test the trained model:
	// Test data with finer resolution:
	float startIntervalTest = -0.99f; float stepSizeTest = 0.03f; int numValuesTest = 67; Mat y_ts(numValuesTest, 2, CV_32FC1); // sin and erf: [-0.99:0.03:0.99]
	{
		vector<float> values(numValuesTest);
		strided_iota(std::begin(values), std::next(std::begin(values), numValuesTest), startIntervalTest, stepSizeTest);
		for (auto r = 0; r < y_ts.rows; ++r) {
			y_ts.at<float>(r, 0) = values[r];
			y_ts.at<float>(r, 1) = values[r];
		}
	}
	Mat x_ts_gt(numValuesTest, 2, CV_32FC1); // Will be the inverse of y_ts
	{
		for (auto r = 0; r < x_ts_gt.rows; ++r) {
			x_ts_gt.at<float>(r, 0) = h_sin_inv(y_ts.at<float>(r, 0));
			x_ts_gt.at<float>(r, 1) = h_erf_inv(y_ts.at<float>(r, 1));
		}
	}
	Mat x0_ts = 0.5f * Mat::ones(numValuesTest, 2, CV_32FC1); // fixed initialization x0 = c = 0.5.

	predictions = sdo.test(x0_ts, y_ts, h);
	double testingResidual = cv::norm(predictions, x_ts_gt, cv::NORM_L2) / cv::norm(x_ts_gt, cv::NORM_L2);
	ASSERT_NEAR(0.0024807, testingResidual, 0.0000021);
}
