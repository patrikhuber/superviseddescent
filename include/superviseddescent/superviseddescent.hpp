/*
 * superviseddescent: A C++11 implementation of the supervised descent
 *                    optimisation method
 * File: superviseddescent/superviseddescent.hpp
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
#pragma once

#ifndef SUPERVISEDDESCENT_HPP_
#define SUPERVISEDDESCENT_HPP_

#include "superviseddescent/utils/ThreadPool.h"

#include "opencv2/core/core.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/serialization/vector.hpp"

/**
 * The main namespace of the supervised descent library.
 *
 * superviseddescent is a C++11 implementation of the supervised descent
 * method, which is a generic algorithm to perform optimisation of arbitrary
 * functions. The basic idea is to learn the gradient direction of a function
 * from data using a series of regressors. As the gradient direction is learned
 * directly from data, the function does not have to be differentiable.
 *
 * The theory is based on the idea of _Supervised Descent Method and Its
 * Applications to Face Alignment_, from X. Xiong & F. De la Torre, CVPR 2013.
 */
namespace superviseddescent {

/**
 * A callback function that gets called after each training of a regressor.
 * This is the default function that performs nothing.
 *
 * @param[in] currentPredictions Predictions of the current regressor.
 */
inline void noEval(const cv::Mat& currentPredictions)
{
};


/**
 * The heart of the library: The main class that performs learning of the
 * gradient, testing and predicting of values.
 *
 * The class takes a \p RegressorType as template parameter to specify a
 * learning algorithm to use. As an example, use a LinearRegressor from
 * regressors.hpp. To use your own learning algorithm, implement the
 * functions of the abstract class Regressor.
 */
template<class RegressorType>
class SupervisedDescentOptimiser
{
public:
	/**
	 * Construct an empty optimiser, with no regressors.
	 *
	 * We allow to create an empty SupervisedDescentOptimiser to
	 * facilitate serialisation.
	 */
	SupervisedDescentOptimiser() = default;

	/**
	 * Construct an optimiser with one or several regressors in series.
	 *
	 * @param[in] regressors One or several regressors.
	 */
	SupervisedDescentOptimiser(std::vector<RegressorType> regressors) : regressors(std::move(regressors))
	{
	};

	/**
	 * Train the regressor model from the given training data.
	 *
	 * The function takes as input a set of groundtruth parameters \p x, initialisation 
	 * values for these parameters \p x0, an optional set of templates \p y (see
	 * description below) and a function \p h that is applied to the parameters.
	 * 
	 * \p y can either be given or set to empty (\p =cv::Mat()); There are two cases:\n
	 *  1. y is known at testing time, i.e. a fixed template. For example pose estimation using given landmarks (=y).
	 *  2. y is not known at testing time. For example landmark detection, where we don't know y at testing (the e.g. HoG values are different for each testing face).
	 *
	 * Examples for the parameters:\n
	 *  1. x = the 6 DOF pose parameters (R, t);
	 *     y = 2D landmark positions;
	 *     x0 = c;
	 *     h = projection from 3D model to 2D, using current params x.
	 *  2. x = groundtruth landmarks - the final location we want to be, i.e. our parameters;
	 *     no y in both training and test;
	 *     x0 = landmark locations after e.g. initialisation with a facedetector;
	 *     h = HoG feature extraction at the current landmark locations x.
	 *
	 * \p h is a function that takes one training sample (a cv::Mat row vector) and returns a cv::Mat row-vector.
	 *
	 * This function calls a default (no-op) callback function after training of each regressor. To specify a callback function, see the overload train(cv::Mat x, cv::Mat x0, cv::Mat y, H h, OnTrainingEpochCallback onTrainingEpochCallback).
	 *
	 * @param[in] x A matrix of ground truth values of parameters, with each row being one training example. These are the parameters we want to learn.
	 * @param[in] x0 Initialisation values for the parameters (x). Constant (x0 = c) in case a) and variable in case b).
	 * @param[in] y An optional matrix of template values. See description above.
	 * @param[in] h The projection function that projects the given x (parameter) values to the space of the y values. Could be a simple function like sin(x), a projection from 3D to 2D space, or a HoG feature transform.
	 */	
	template<class H>
	void train(cv::Mat x, cv::Mat x0, cv::Mat y, H h)
	{
		return train(x, x0, y, h, noEval);
	};

	/**
	 * Identical to train(cv::Mat x, cv::Mat x0, cv::Mat y, H h), but allows to
	 * specify a callback function that gets called with the current prediction
	 * after every regressor.
	 *
	 * The signature of the callback function must take a cv::Mat with the current
	 * predictions and can capture any additional variables from the surrounding context.
	 * For example, to print the normalised least squares residual between
	 * \p groundtruth and the current predictions:
	 * \code
	 * auto printResidual = [&groundtruth](const cv::Mat& currentPredictions) {
	 *	std::cout << cv::norm(currentPredictions, groundtruth, cv::NORM_L2) / cv::norm(groundtruth, cv::NORM_L2) << std::endl;
	 * };
	 * \endcode
	 *
	 * @copydoc train(cv::Mat x, cv::Mat x0, cv::Mat y, H h).
	 *
	 * @param[in] onTrainingEpochCallback A callback function that gets called after the training of each individual regressor.
	 */
	template<class H, class OnTrainingEpochCallback>
	void train(cv::Mat x, cv::Mat x0, cv::Mat y, H h, OnTrainingEpochCallback onTrainingEpochCallback)
	{
		using cv::Mat;
		Mat currentX = x0;
		for (size_t regressorLevel = 0; regressorLevel < regressors.size(); ++regressorLevel) {
			// 1) Project current parameters x to feature space:
			// Enqueue all tasks in a thread pool:
			auto concurentThreadsSupported = std::thread::hardware_concurrency();
			if (concurentThreadsSupported == 0) {
				concurentThreadsSupported = 4;
			}
			utils::ThreadPool threadPool(concurentThreadsSupported);
			std::vector<std::future<typename std::result_of<H(Mat, size_t, int)>::type>> results; // will be float or Mat. I might remove float for the sake of code clarity, as it's only useful for very simple examples.
			results.reserve(currentX.rows);
			for (int sampleIndex = 0; sampleIndex < currentX.rows; ++sampleIndex) {
				results.emplace_back(
					threadPool.enqueue(h, currentX.row(sampleIndex), regressorLevel, sampleIndex)
				);
			}
			// Gather the results from all threads and store the features:
			Mat features;
			for (auto&& result : results) {
				features.push_back(result.get());
			}
			// Set the observed values, depending on if a template y is used:
			Mat observedValues;
			if (y.empty()) { // unknown y training case
				observedValues = features;
			}
			else { // known y
				observedValues = features - y;
			}
			Mat b = currentX - x;
			// 2) Learn using that data:
			regressors[regressorLevel].learn(observedValues, b);
			// 3) Apply the learned regressor and use the predictions to learn the next regressor in next loop iteration:
			Mat x_k; // x_k = currentX - R * (h(currentX) - y):
			for (int sampleIndex = 0; sampleIndex < currentX.rows; ++sampleIndex) {
				// No need to re-extract the features, we already did so in step 1)
				x_k.push_back(Mat(currentX.row(sampleIndex) - regressors[regressorLevel].predict(observedValues.row(sampleIndex))));
			}
			currentX = x_k;
			onTrainingEpochCallback(currentX);
		}
	};

	/**
	 * Tests the learned regressor model with the given test data.
	 *
	 * The test data should be specified as one row per example.
	 *
	 * \p y can either be given or set to empty (\p =cv::Mat()), according to what was learned during training.
	 *
	 * For a detailed explanation of the parameters, see train(cv::Mat x, cv::Mat x0, cv::Mat y, H h).
	 *
	 * Calls the testing with a default (no-op) callback function. To specify a callback function, see the overload test(cv::Mat x0, cv::Mat y, H h, OnRegressorIterationCallback onRegressorIterationCallback).
	 *
	 * @param[in] x0 Initialisation values for the parameters (x). Constant (x0 = c) in case a) and variable in case b).
	 * @param[in] y An optional matrix of template values. See description above.
	 * @param[in] h The projection function that projects the given x (parameter) values to the space of the y values. Could be a simple function like sin(x), a projection from 3D to 2D space, or a HoG feature transform.
	 * @return Returns the final prediction of all given test examples.
	 */
	template<class H>
	cv::Mat test(cv::Mat x0, cv::Mat y, H h)
	{
		return test(x0, y, h, noEval);
	};

	/**
	 * Identical to test(cv::Mat x0, cv::Mat y, H h), but allows to specify a
	 * callback function that gets called with the current prediction
	 * after applying each regressor.
	 *
	 * The signature of the callback function must take a cv::Mat with the current
	 * predictions and can capture any additional variables from the surrounding context.
	 * For example, to print the normalised least squares residual between
	 * \p groundtruth and the current predictions:
	 * \code
	 * auto printResidual = [&groundtruth](const cv::Mat& currentPredictions) {
	 *	std::cout << cv::norm(currentPredictions, groundtruth, cv::NORM_L2) / cv::norm(groundtruth, cv::NORM_L2) << std::endl;
	 * };
	 * \endcode
	 *
	 * @copydoc test(cv::Mat x0, cv::Mat y, H h).
	 *
	 * @param[in] onRegressorIterationCallback A callback function that gets called after each applied regressor, with the current prediction result.
	 */
	template<class H, class OnRegressorIterationCallback>
	cv::Mat test(cv::Mat x0, cv::Mat y, H h, OnRegressorIterationCallback onRegressorIterationCallback)
	{
		using cv::Mat;
		Mat currentX = x0;
		for (size_t regressorLevel = 0; regressorLevel < regressors.size(); ++regressorLevel) {
			// Enqueue all tasks in a thread pool:
			auto concurentThreadsSupported = std::thread::hardware_concurrency();
			if (concurentThreadsSupported == 0) {
				concurentThreadsSupported = 4;
			}
			utils::ThreadPool threadPool(concurentThreadsSupported);
			std::vector<std::future<typename std::result_of<H(Mat, size_t, int)>::type>> results; // will be float or Mat. I might remove float for the sake of code clarity, as it's only useful for very simple examples.
			results.reserve(currentX.rows);
			for (int sampleIndex = 0; sampleIndex < currentX.rows; ++sampleIndex) {
				results.emplace_back(
					threadPool.enqueue(h, currentX.row(sampleIndex), regressorLevel, sampleIndex)
					);
			}
			// Gather the results from all threads and store the features:
			Mat features;
			for (auto&& result : results) {
				features.push_back(result.get());
			}

			Mat observedValues;
			if (y.empty()) { // unknown y training case
				observedValues = features;
			}
			else { // known y
				observedValues = features - y;
			}
			Mat x_k;
			// Calculate x_k = currentX - R * (h(currentX) - y):
			for (int sampleIndex = 0; sampleIndex < currentX.rows; ++sampleIndex) {
				x_k.push_back(Mat(currentX.row(sampleIndex) - regressors[regressorLevel].predict(observedValues.row(sampleIndex)))); // we need Mat() because the subtraction yields a (non-persistent) MatExpr
			}
			currentX = x_k;
			onRegressorIterationCallback(currentX);
		}
		return currentX; // Return the final predictions
	};

	/**
	 * Predicts the result value for a single example, using the learned
	 * regressors.
	 *
	 * The input matrices should only contain one row (i.e. one training example).
	 *
	 * \p y can either be given or set to empty (\p =cv::Mat()), according to what was learned during training.
	 *
	 * For a detailed explanation of the parameters, see train(cv::Mat x, cv::Mat x0, cv::Mat y, H h).
	 *
	 * @param[in] x0 Initialisation values for the parameters (x). Constant (x0 = c) in case a) and variable in case b).
	 * @param[in] y An optional matrix of template values. See description above.
	 * @param[in] h The projection function that projects the given x (parameter) values to the space of the y values. Could be a simple function like sin(x), a projection from 3D to 2D space, or a HoG feature transform.
	 * @return Returns the predicted parameters for the given test example.
	 */	
	template<class H>
	cv::Mat predict(cv::Mat x0, cv::Mat y, H h)
	{
		using cv::Mat;
		Mat currentX = x0;
		for (size_t r = 0; r < regressors.size(); ++r) {
			// calculate x_k = currentX - R * (h(currentX) - y):
			Mat observedValues;
			if (y.empty()) { // unknown y training case
				observedValues = h(currentX, r);
			}
			else { // known y
				observedValues = h(currentX, r) - y;
			}
			Mat x_k = currentX - regressors[r].predict(observedValues);
			currentX = x_k;
		}
		return currentX;
	};

private:
	std::vector<RegressorType> regressors; ///< A series of learned regressors.

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
		ar & regressors;
	}
};

} /* namespace superviseddescent */
#endif /* SUPERVISEDDESCENT_HPP_ */
