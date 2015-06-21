#include "gtest/gtest.h"

#include "superviseddescent/regressors.hpp"

#include "opencv2/core/core.hpp"

using namespace superviseddescent;

/*
TEST(LinearRegressor, NDimOneExampleLearningNotInvertible) {
	using cv::Mat;
	Mat data = Mat::ones(1, 2, CV_32FC1);
	Mat labels = Mat::ones(1, 1, CV_32FC1);
	// A simple case of a singular matrix. Yields infinitely many possible results.
	LinearRegressor<> lr;
	bool isInvertible = lr.learn(data, labels);
	ASSERT_EQ(false, isInvertible);
}
*/

TEST(LinearRegressor, NDimOneExampleLearningRegularisation) {
	using cv::Mat;
	Regulariser r(Regulariser::RegularisationType::Manual, 1.0f, true); // no bias, so regularise every data-row
	Mat data = Mat::ones(1, 2, CV_32FC1);
	Mat labels = Mat::ones(1, 1, CV_32FC1);
	// This case becomes solvable with regularisation
	LinearRegressor<> lr(r);
	bool isInvertible = lr.learn(data, labels);
	EXPECT_FLOAT_EQ(1.0f/3.0f, lr.x.at<float>(0)) << "Expected the learned x_0 to be 1.0f/3.0f";
	EXPECT_FLOAT_EQ(1.0f/3.0f, lr.x.at<float>(1)) << "Expected the learned x_1 to be 1.0f/3.0f";
	ASSERT_EQ(true, isInvertible);
}

// We can't construct an invertible 2D example with 1 training point
TEST(LinearRegressor, NDimTwoExamplesLearning) {
	using cv::Mat;
	Mat data = Mat::ones(2, 2, CV_32FC1);
	data.at<float>(0) = 0.0f; // data = [0 1; 1 1]
	Mat labels = Mat::ones(2, 1, CV_32FC1); // The label can also be multi-dim. More test cases?
	labels.at<float>(0) = 0.0f; // labels = [0; 1]
	LinearRegressor<> lr;
	bool isInvertible = lr.learn(data, labels);
	EXPECT_EQ(true, isInvertible);
	EXPECT_FLOAT_EQ(1.0f, lr.x.at<float>(0)) << "Expected the learned x_0 to be 1.0f";
	ASSERT_FLOAT_EQ(0.0f, lr.x.at<float>(1)) << "Expected the learned x_1 to be 0.0f";
}

TEST(LinearRegressor, NDimTwoExamplesPrediction) {
	using cv::Mat;
	// Note/Todo: Load from filesystem, or from memory-bytes?
	Mat data = Mat::ones(2, 2, CV_32FC1);
	data.at<float>(0) = 0.0f; // data = [0 1; 1 1]
	Mat labels = Mat::ones(2, 1, CV_32FC1); // The label can also be multi-dim. More test cases?
	labels.at<float>(0) = 0.0f; // labels = [0; 1]
	LinearRegressor<> lr;
	lr.learn(data, labels);

	// Test starts here:
	Mat test = 2.0f * Mat::ones(1, 2, CV_32FC1);
	Mat prediction = lr.predict(test);
	ASSERT_FLOAT_EQ(2.0f, prediction.at<float>(0)) << "Expected the prediction to be 2.0f";
}

TEST(LinearRegressor, NDimTwoExamplesTestingResidual) {
	using cv::Mat;
	// Note/Todo: Load from filesystem, or from memory-bytes?
	Mat data = Mat::ones(2, 2, CV_32FC1);
	data.at<float>(0) = 0.0f; // data = [0 1; 1 1]
	Mat labels = Mat::ones(2, 1, CV_32FC1); // The label can also be multi-dim. More test cases?
	labels.at<float>(0) = 0.0f; // labels = [0; 1]
	LinearRegressor<> lr;
	lr.learn(data, labels);

	// Test starts here:
	Mat test(3, 2, CV_32FC1);
	test.at<float>(0, 0) = 0.0f;
	test.at<float>(0, 1) = 2.0f;
	test.at<float>(1, 0) = 2.0f;
	test.at<float>(1, 1) = 1.0f;
	test.at<float>(2, 0) = 2.0f;
	test.at<float>(2, 1) = 1.0f; // test = [ 0 2; 2 1; 2 1]
	Mat groundtruth(3, 1, CV_32FC1);
	groundtruth.at<float>(0) = 0.0f;
	groundtruth.at<float>(1) = 2.0f;
	groundtruth.at<float>(2) = -1.0f; // gt = [ 0 (error 0), 2 (error 0), -1 (error -3)]
	double residual = lr.test(test, groundtruth);
	ASSERT_NEAR(1.3416407, residual, 0.0000001);
}

TEST(LinearRegressor, NDimTwoExamplesNDimYLearning) {
	using cv::Mat;
	Mat data = Mat::ones(2, 2, CV_32FC1);
	data.at<float>(0) = 0.0f; // data = [0 1; 1 1]
	Mat labels = Mat::ones(2, 2, CV_32FC1); // The label can also be multi-dim. More test cases?
	labels.at<float>(0) = 0.0f; // labels = [0 1; 1 1]
	LinearRegressor<> lr;
	bool isInvertible = lr.learn(data, labels);
	EXPECT_EQ(true, isInvertible);
	EXPECT_FLOAT_EQ(1.0f, lr.x.at<float>(0, 0)) << "Expected the learned x_0_0 to be 1.0f"; // Every col is a learned regressor for a label
	EXPECT_FLOAT_EQ(0.0f, lr.x.at<float>(1, 0)) << "Expected the learned x_1_0 to be 0.0f";
	EXPECT_FLOAT_EQ(0.0f, lr.x.at<float>(0, 1)) << "Expected the learned x_0_1 to be 0.0f";
	ASSERT_FLOAT_EQ(1.0f, lr.x.at<float>(1, 1)) << "Expected the learned x_1_1 to be 1.0f";
}

TEST(LinearRegressor, NDimTwoExamplesNDimYPrediction) {
	using cv::Mat;
	Mat data = Mat::ones(2, 2, CV_32FC1);
	data.at<float>(0) = 0.0f; // data = [0 1; 1 1]
	Mat labels = Mat::ones(2, 2, CV_32FC1); // The label can also be multi-dim. More test cases?
	labels.at<float>(0) = 0.0f; // labels = [0 1; 1 1]
	LinearRegressor<> lr;
	bool isInvertible = lr.learn(data, labels);
	EXPECT_EQ(true, isInvertible);

	// Test starts here:
	Mat test = Mat::ones(1, 2, CV_32FC1);
	test.at<float>(1) = 2.0f;
	Mat prediction = lr.predict(test);
	EXPECT_FLOAT_EQ(1.0f, prediction.at<float>(0)) << "Expected the predicted y_0 to be 1.0f";
	ASSERT_FLOAT_EQ(2.0f, prediction.at<float>(1)) << "Expected the predicted y_1 to be 2.0f";
}

TEST(LinearRegressor, NDimTwoExamplesNDimYTestingResidual) {
	using cv::Mat;
	Mat data = Mat::ones(2, 2, CV_32FC1);
	data.at<float>(0) = 0.0f; // data = [0 1; 1 1]
	Mat labels = Mat::ones(2, 2, CV_32FC1); // The label can also be multi-dim. More test cases?
	labels.at<float>(0) = 0.0f; // labels = [0 1; 1 1]
	LinearRegressor<> lr;
	bool isInvertible = lr.learn(data, labels);
	EXPECT_EQ(true, isInvertible);

	// Test starts here:
	Mat test(3, 2, CV_32FC1);
	test.at<float>(0, 0) = 0.0f;
	test.at<float>(0, 1) = 2.0f;
	test.at<float>(1, 0) = 2.0f;
	test.at<float>(1, 1) = 1.0f;
	test.at<float>(2, 0) = 2.0f;
	test.at<float>(2, 1) = 1.0f; // test = [ 0 2; 2 1; 2 1]
	Mat groundtruth(3, 2, CV_32FC1);
	groundtruth.at<float>(0, 0) = 0.0f;
	groundtruth.at<float>(1, 0) = 2.0f;
	groundtruth.at<float>(2, 0) = -1.0f;
	groundtruth.at<float>(0, 1) = 0.0f;
	groundtruth.at<float>(1, 1) = 4.0f;
	groundtruth.at<float>(2, 1) = -2.0f;
	double residual = lr.test(test, groundtruth);
	ASSERT_NEAR(1.11355285, residual, 0.00000004);
}

TEST(LinearRegressor, NDimManyExamplesNDimY) {
	using cv::Mat;
	// An invertible example, constructed from Matlab
	Mat data = (cv::Mat_<float>(5, 3) << 1.0f, 4.0f, 2.0f, 4.0f, 9.0f, 1.0f, 6.0f, 5.0f, 2.0f, 0.0f, 6.0f, 2.0f, 6.0f, 1.0f, 9.0f);
	Mat labels = (cv::Mat_<float>(5, 2) << 1.0f, 1.0f, 2.0f, 5.0f, 3.0f, -2.0f, 0.0f, 5.0f, 6.0f, 3.0f);
	LinearRegressor<> lr;
	bool isInvertible = lr.learn(data, labels);
	EXPECT_EQ(true, isInvertible);
	EXPECT_NEAR(0.489539f, lr.x.at<float>(0, 0), 0.000002); // Every col is a learned regressor for a label
	EXPECT_NEAR(-0.06608297f, lr.x.at<float>(1, 0), 0.00000003);
	EXPECT_FLOAT_EQ(0.339629412f, lr.x.at<float>(2, 0));
	EXPECT_FLOAT_EQ(-0.833899379f, lr.x.at<float>(0, 1));
	EXPECT_FLOAT_EQ(0.626753688f, lr.x.at<float>(1, 1));
	EXPECT_FLOAT_EQ(0.744218946f, lr.x.at<float>(2, 1));

	// Testing:
	Mat test = (cv::Mat_<float>(3, 3) << 2.0f, 6.0f, 5.0f, 2.9f, -11.3, 6.0f, -2.0f, -8.438f, 3.3f);
	Mat groundtruth = (cv::Mat_<float>(3, 2) << 2.2807f, 5.8138f, 4.2042f, -5.0353f, 0.6993f, -1.1648f);
	double residual = lr.test(test, groundtruth);
	ASSERT_LE(residual, 0.000006) << "Expected the residual to be smaller than "; // we could relax that to 0.00001 or even larger, but for now, I want to be pedantic
}

TEST(LinearRegressor, NDimManyExamplesNDimYRegularisation) {
	using cv::Mat;
	// An invertible example, constructed from Matlab
	Mat data = (cv::Mat_<float>(5, 3) << 1.0f, 4.0f, 2.0f, 4.0f, 9.0f, 1.0f, 6.0f, 5.0f, 2.0f, 0.0f, 6.0f, 2.0f, 6.0f, 1.0f, 9.0f);
	Mat labels = (cv::Mat_<float>(5, 2) << 1.0f, 1.0f, 2.0f, 5.0f, 3.0f, -2.0f, 0.0f, 5.0f, 6.0f, 3.0f);
	Regulariser r(Regulariser::RegularisationType::Manual, 50.0f, true); // no bias, so regularise every data-row
	LinearRegressor<> lr(r);
	bool isInvertible = lr.learn(data, labels);
	EXPECT_EQ(true, isInvertible);
	EXPECT_FLOAT_EQ(0.282755911f, lr.x.at<float>(0, 0)) << "Expected the learned x_0_0 to be different"; // Every col is a learned regressor for a label
	EXPECT_NEAR(0.03607957f, lr.x.at<float>(1, 0), 0.00000002) << "Expected the learned x_1_0 to be different";
	EXPECT_FLOAT_EQ(0.291039944f, lr.x.at<float>(2, 0)) << "Expected the learned x_2_0 to be different";
	EXPECT_NEAR(-0.0989616f, lr.x.at<float>(0, 1), 0.0000001) << "Expected the learned x_0_1 to be different";
	EXPECT_FLOAT_EQ(0.330635577f, lr.x.at<float>(1, 1)) << "Expected the learned x_1_1 to be different";
	EXPECT_FLOAT_EQ(0.217046738f, lr.x.at<float>(2, 1)) << "Expected the learned x_2_1 to be different";

	// Testing:
	Mat test = (cv::Mat_<float>(3, 3) << 2.0f, 6.0f, 5.0f, 2.9f, -11.3, 6.0f, -2.0f, -8.438f, 3.3f);
	Mat groundtruth = (cv::Mat_<float>(3, 2) << 2.2372f, 2.8711f, 2.1585f, -2.7209f, 0.0905f, -1.8757f);
	double residual = lr.test(test, groundtruth);
	ASSERT_LE(residual, 0.000011) << "Expected the residual to be smaller than "; // we could relax that a bit
}

TEST(LinearRegressor, NDimManyExamplesNDimYBias) {
	using cv::Mat;
	// An invertible example, constructed from Matlab
	Mat data = (cv::Mat_<float>(5, 3) << 1.0f, 4.0f, 2.0f, 4.0f, 9.0f, 1.0f, 6.0f, 5.0f, 2.0f, 0.0f, 6.0f, 2.0f, 6.0f, 1.0f, 9.0f);
	Mat labels = (cv::Mat_<float>(5, 2) << 1.0f, 1.0f, 2.0f, 5.0f, 3.0f, -2.0f, 0.0f, 5.0f, 6.0f, 3.0f);
	LinearRegressor<> lr;
	Mat biasColumn = Mat::ones(data.rows, 1, CV_32FC1);
	cv::hconcat(data, biasColumn, data);
	bool isInvertible = lr.learn(data, labels);
	EXPECT_EQ(true, isInvertible);
	EXPECT_NEAR(0.485009f, lr.x.at<float>(0, 0), 0.000001) << "Expected the learned x_0_0 to be different"; // Every col is a learned regressor for a label
	EXPECT_NEAR(0.012218f, lr.x.at<float>(1, 0), 0.000002) << "Expected the learned x_1_0 to be different";
	EXPECT_NEAR(0.407823f, lr.x.at<float>(2, 0), 0.000002) << "Expected the learned x_2_0 to be different";
	EXPECT_NEAR(-0.61515f, lr.x.at<float>(3, 0), 0.00001) << "Expected the learned x_3_0 to be different";
	EXPECT_NEAR(-0.894791f, lr.x.at<float>(0, 1), 0.000001) << "Expected the learned x_0_1 to be different";
	EXPECT_NEAR(1.679203f, lr.x.at<float>(1, 1), 0.000003) << "Expected the learned x_1_1 to be different";
	EXPECT_NEAR(1.660814f, lr.x.at<float>(2, 1), 0.000002) << "Expected the learned x_2_1 to be different";
	EXPECT_NEAR(-8.26833f, lr.x.at<float>(3, 1), 0.00002) << "Expected the learned x_3_1 to be different";

	// Testing:
	Mat test = (cv::Mat_<float>(3, 3) << 2.0f, 6.0f, 5.0f, 2.9f, -11.3, 6.0f, -2.0f, -8.438f, 3.3f);
	Mat biasColumnTest = Mat::ones(test.rows, 1, CV_32FC1);
	cv::hconcat(test, biasColumnTest, test);
	Mat groundtruth = (cv::Mat_<float>(3, 2) << 2.4673f, 8.3214f, 3.1002f, -19.8734f, -0.3425f, -15.1672f);
	double residual = lr.test(test, groundtruth);
	ASSERT_LE(residual, 0.000006); // we could relax that a bit
}

// Actually we could test the Regulariser separately, no need to have separate unit tests.
TEST(LinearRegressor, NDimManyExamplesNDimYBiasRegularisation) {
	using cv::Mat;
	// An invertible example, constructed from Matlab
	Mat data = (cv::Mat_<float>(5, 3) << 1.0f, 4.0f, 2.0f, 4.0f, 9.0f, 1.0f, 6.0f, 5.0f, 2.0f, 0.0f, 6.0f, 2.0f, 6.0f, 1.0f, 9.0f);
	Mat labels = (cv::Mat_<float>(5, 2) << 1.0f, 1.0f, 2.0f, 5.0f, 3.0f, -2.0f, 0.0f, 5.0f, 6.0f, 3.0f);
	Regulariser r(Regulariser::RegularisationType::Manual, 50.0f, true); // regularise the bias as well
	LinearRegressor<> lr(r);
	Mat biasColumn = Mat::ones(data.rows, 1, CV_32FC1);
	cv::hconcat(data, biasColumn, data);
	bool isInvertible = lr.learn(data, labels);
	EXPECT_EQ(true, isInvertible);
	EXPECT_NEAR(0.2814246f, lr.x.at<float>(0, 0), 0.0000002) << "Expected the learned x_0_0 to be different"; // Every col is a learned regressor for a label
	EXPECT_NEAR(0.03317654f, lr.x.at<float>(1, 0), 00000003) << "Expected the learned x_1_0 to be different";
	EXPECT_FLOAT_EQ(0.289116770f, lr.x.at<float>(2, 0)) << "Expected the learned x_2_0 to be different";
	EXPECT_FLOAT_EQ(0.0320090912f, lr.x.at<float>(3, 0)) << "Expected the learned x_3_0 to be different";
	EXPECT_NEAR(-0.1005448f, lr.x.at<float>(0, 1), 0.0000001) << "Expected the learned x_0_1 to be different";
	EXPECT_FLOAT_EQ(0.327183396f, lr.x.at<float>(1, 1)) << "Expected the learned x_1_1 to be different";
	EXPECT_FLOAT_EQ(0.214759737f, lr.x.at<float>(2, 1)) << "Expected the learned x_2_1 to be different";
	EXPECT_NEAR(0.03806401f, lr.x.at<float>(3, 1), 0.00000002) << "Expected the learned x_3_1 to be different";

	// Testing:
	Mat test = (cv::Mat_<float>(3, 3) << 2.0f, 6.0f, 5.0f, 2.9f, -11.3, 6.0f, -2.0f, -8.438f, 3.3f);
	Mat biasColumnTest = Mat::ones(test.rows, 1, CV_32FC1);
	cv::hconcat(test, biasColumnTest, test);
	Mat groundtruth = (cv::Mat_<float>(3, 2) << 2.2395f, 2.8739f, 2.2079f, -2.6621f, 0.1433f, -1.8129f);
	double residual = lr.test(test, groundtruth);
	ASSERT_LE(residual, 0.000012) << "Expected the residual to be smaller than "; // we could relax that a bit
}

TEST(LinearRegressor, NDimManyExamplesNDimYBiasRegularisationButNotBias) {
	using cv::Mat;
	// An invertible example, constructed from Matlab
	Mat data = (cv::Mat_<float>(5, 3) << 1.0f, 4.0f, 2.0f, 4.0f, 9.0f, 1.0f, 6.0f, 5.0f, 2.0f, 0.0f, 6.0f, 2.0f, 6.0f, 1.0f, 9.0f);
	Mat labels = (cv::Mat_<float>(5, 2) << 1.0f, 1.0f, 2.0f, 5.0f, 3.0f, -2.0f, 0.0f, 5.0f, 6.0f, 3.0f);
	Regulariser r(Regulariser::RegularisationType::Manual, 50.0f, false); // don't regularise the bias
	LinearRegressor<> lr(r);
	Mat biasColumn = Mat::ones(data.rows, 1, CV_32FC1);
	cv::hconcat(data, biasColumn, data);
	bool isInvertible = lr.learn(data, labels);
	EXPECT_EQ(true, isInvertible);
	EXPECT_NEAR(0.2188783f, lr.x.at<float>(0, 0), 0.0000002) << "Expected the learned x_0_0 to be different"; // Every col is a learned regressor for a label
	EXPECT_NEAR(-0.1032114f, lr.x.at<float>(1, 0), 0.0000001) << "Expected the learned x_1_0 to be different";
	EXPECT_NEAR(0.1987606f, lr.x.at<float>(2, 0), 0.0000002) << "Expected the learned x_2_0 to be different";
	EXPECT_FLOAT_EQ(1.53583705f, lr.x.at<float>(3, 0)) << "Expected the learned x_3_0 to be different";
	EXPECT_FLOAT_EQ(-0.174922630f, lr.x.at<float>(0, 1)) << "Expected the learned x_0_1 to be different";
	EXPECT_FLOAT_EQ(0.164996058f, lr.x.at<float>(1, 1)) << "Expected the learned x_1_1 to be different";
	EXPECT_NEAR(0.1073116f, lr.x.at<float>(2, 1), 0.0000001) << "Expected the learned x_2_1 to be different";
	EXPECT_FLOAT_EQ(1.82635951f, lr.x.at<float>(3, 1)) << "Expected the learned x_3_1 to be different";

	// Testing:
	Mat test = (cv::Mat_<float>(3, 3) << 2.0f, 6.0f, 5.0f, 2.9f, -11.3, 6.0f, -2.0f, -8.438f, 3.3f);
	Mat biasColumnTest = Mat::ones(test.rows, 1, CV_32FC1);
	cv::hconcat(test, biasColumnTest, test);
	Mat groundtruth = (cv::Mat_<float>(3, 2) << 2.3481f, 3.0030f, 4.5294f, 0.0985f, 2.6249f, 1.1381f);
	double residual = lr.test(test, groundtruth);
	ASSERT_LE(residual, 0.000011) << "Expected the residual to be smaller than "; // we could relax that a bit
}
