/*
 * superviseddescent: A C++11 implementation of the supervised descent
 *                    optimisation method
 * File: superviseddescent/regressors.hpp
 *
 * Copyright 2014 Patrik Huber
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

#ifndef REGRESSORS_HPP_
#define REGRESSORS_HPP_

#include "matserialisation.hpp"

#include "opencv2/core/core.hpp"
#include "Eigen/Dense"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/serialization/serialization.hpp"

#include <cassert>
#include <iostream>

namespace superviseddescent {

/**
 * Abstract base class for regressor-like learning algorithms to be used with
 * the SupervisedDescentOptimiser.
 *
 * Classes that implement this minimal set of functions can be used with the
 * SupervisedDescentOptimiser.
 */
class Regressor
{
public:
	virtual ~Regressor() {};

	/**
	 * Learning function that takes a matrix of data, with one example per
	 * row, and a corresponding matrix of labels, with one or multiple labels
	 * per training datum.
	 *
	 * @param[in] data Training data matrix, one example per row.
	 * @param[in] labels Labels corresponding to the training data.
	 * @return Returns whether the learning was successful.
	 */
	virtual bool learn(cv::Mat data, cv::Mat labels) = 0;

	/**
	 * Test the learned regressor, using the given data (one row for every
	 * element) and corresponding labels. Calculates the normalised least squares
	 * residual \f$ \frac{\|\mathbf{prediction}-\mathbf{labels}\|}{\|\mathbf{labels}\|} \f$.
	 *
	 * @param[in] data Test data matrix.
	 * @param[in] labels Corresponding label information.
	 * @return The normalised least squares residual.
	 */
	virtual double test(cv::Mat data, cv::Mat labels) = 0;

	/**
	 * Predicts the regressed value for one given sample.
	 *
	 * @param[in] values One data point as a row vector.
	 * @return The predicted value(s).
	 */
	virtual cv::Mat predict(cv::Mat values) = 0;
};

/**
 * Regulariser class for the LinearRegressor.
 *
 * Produces a diagonal matrix with a regularisation value (lambda) on the 
 * diagonal that can be added to the data matrix. Lambda can either be fixed
 * or calculated from the data matrix.
 *
 */
class Regulariser
{
public:
	/**
	 * The method to calculate the regularisation factor lambda.
	 */
	enum class RegularisationType
	{
		Manual, ///< Use the given param value as lambda.
		MatrixNorm, ///< Use \f$ \text{lambda} = \text{param} * \frac{\|\text{data}\|_2}{\text{numTrainingElements}} \f$ A suitable default for _param_ suggested by the SDM authors is 0.5.
	};

	/**
	 * Create a new regulariser. Created with default parameters, it will not
	 * do any regularisation. Regulariser::RegularisationType can be used to specify the
	 * choice of the regularisation parameter lambda.
	 *
	 * _regulariseLastRow_ is useful to specify the regularisation behaviour in
	 * the case the last row of the data matrix contains an affine (offset or
	 * bias) component. In that case, you might not want to regularise it (or
	 * maybe you do).
	 *
	 * @param[in] regularisationType Specifies how to calculate lambda.
	 * @param[in] param Lambda, or a factor, depending on regularisationType.
	 * @param[in] regulariseLastRow Specifies if the last row should be regularised.
	 */
	Regulariser(RegularisationType regularisationType = RegularisationType::Manual, float param = 0.0f, bool regulariseLastRow = true) : regularisationType(regularisationType), lambda(param), regulariseLastRow(regulariseLastRow)
	{
	};

	/**
	 * Calculates a diagonal regularisation matrix, given the RegularisationType
	 * and param of the regulariser. It will have the same dimensions as the
	 * given data matrix.
	 *
	 * @param[in] data Data matrix that might be used to calculate an automatic value for lambda.
	 * @param[in] numTrainingElements Number of training elements.
	 * @return Returns a diagonal regularisation matrix with the same dimensions as the given data matrix.
	 */
	cv::Mat getMatrix(cv::Mat data, int numTrainingElements)
	{
		switch (regularisationType)
		{
		case RegularisationType::Manual:
			// We just take lambda as it was given, no calculation necessary.
			break;
		case RegularisationType::MatrixNorm:
			// The given lambda is the factor we have to multiply the automatic value with:
			lambda = lambda * static_cast<float>(cv::norm(data)) / static_cast<float>(numTrainingElements);
			break;
		default:
			break;
		}

		cv::Mat regulariser = cv::Mat::eye(data.rows, data.cols, CV_32FC1) * lambda;

		if (!regulariseLastRow) {
			// no lambda for the bias:
			regulariser.at<float>(regulariser.rows - 1, regulariser.cols - 1) = 0.0f;
		}
		return regulariser;
	};

private:
	RegularisationType regularisationType; ///< The type of regularisation this regulariser is using.
	float lambda; ///< The parameter for RegularisationType. Can be lambda directly or a factor with which the lambda from MatrixNorm will be multiplied with.
	bool regulariseLastRow; ///< If the last row of data matrix is a bias (offset), then you might want to choose whether it should be regularised as well. Otherwise, just leave it to default (true).

	friend class boost::serialization::access;
	/**
	 * Serialises this class using boost::serialization.
	 *
	 * @param[in] ar The archive to serialise to (or to serialise from).
	 * @param[in] version An optional version argument.
	 */
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & regularisationType;
		ar & lambda;
		ar & regulariseLastRow;
	}
};

/**
 * A simple LinearRegressor that learns coefficients x for the linear relationship
 * \f$ Ax = b \f$. This class handles learning, testing, and predicting single examples.
 *
 * A Regulariser can be specified to make the least squares problem more
 * well-behaved (or invertible, in case it is not).
 *
 * Works with multi-dimensional label data. In that case, the coefficients for
 * each label will be learned independently.
 */
class LinearRegressor : public Regressor
{

public:
	/**
	 * Creates a LinearRegressor with no regularisation.
	 *
	 * @param[in] regulariser A Regulariser to regularise the data matrix. Default means no regularisation.
	 */
	LinearRegressor(Regulariser regulariser = Regulariser()) : x(), regulariser(regulariser)
	{
	};

	/**
	 * Learns a linear predictor from the given data and labels.
	 *
	 * In case the problem is not invertible, the function will return false
	 * and will most likely have learned garbage.
	 *
	 * @param[in] data Training data matrix, one example per row.
	 * @param[in] labels Labels corresponding to the training data.
	 * @return Returns whether \f$ \text{data}^t * \text{data} \f$ was invertible.
	 */
	bool learn(cv::Mat data, cv::Mat labels)
	{
		using cv::Mat;

		Mat AtA = data.t() * data;

		Mat regularisationMatrix = regulariser.getMatrix(AtA, data.rows);
		
		AtA = AtA + regularisationMatrix;
		assert(AtA.isContinuous());
		// Map the cv::Mat data to an Eigen::Matrix and perform a FullPivLU:
		Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> AtA_Eigen(AtA.ptr<float>(), AtA.rows, AtA.cols);
		Eigen::FullPivLU<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> luOfAtA(AtA_Eigen);
		auto rankOfAtA = luOfAtA.rank();
		if (!luOfAtA.isInvertible()) {
			// Eigen will most likely return garbage here (according to their docu anyway). We have a few options:
			// - Increase lambda
			// - Calculate the pseudo-inverse. See: http://eigen.tuxfamily.org/index.php?title=FAQ#Is_there_a_method_to_compute_the_.28Moore-Penrose.29_pseudo_inverse_.3F
			std::cout << "The regularised AtA is not invertible. We continued learning, but Eigen calculates garbage in this case according to their documentation. (The rank is " << std::to_string(rankOfAtA) << ", full rank would be " << std::to_string(AtA_Eigen.rows()) << "). Increase lambda (or use the pseudo-inverse, which is not implemented yet)." << std::endl;
		}
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> AtAInv_Eigen = luOfAtA.inverse();
		// Map the inverted matrix back to a cv::Mat by creating a Mat header:
		Mat AtAInv(static_cast<int>(AtAInv_Eigen.rows()), static_cast<int>(AtAInv_Eigen.cols()), CV_32FC1, AtAInv_Eigen.data());
		
		// x = (AtAReg)^-1 * At * b:
		x = AtAInv * data.t() * labels; // store the result in the member variable
		
		return luOfAtA.isInvertible();
	};

	/**
	 * Test the learned regressor, using the given data (one row for every
	 * element) and corresponding labels. Calculates the normalised least squares
	 * residual \f$ \frac{\|\text{prediction} - \text{labels}\|}{\|\text{labels}\|} \f$.
	 *
	 * @param[in] data Test data matrix.
	 * @param[in] labels Corresponding label information.
	 * @return The normalised least squares residual.
	 */
	double test(cv::Mat data, cv::Mat labels)
	{
		cv::Mat predictions;
		for (int i = 0; i < data.rows; ++i) {
			cv::Mat prediction = predict(data.row(i));
			predictions.push_back(prediction);
		}
		return cv::norm(predictions, labels, cv::NORM_L2) / cv::norm(labels, cv::NORM_L2);
	};

	/**
	 * Predicts the regressed value for one given sample.
	 *
	 * @param[in] values One data point as a row vector.
	 * @return The predicted value(s).
	 */
	cv::Mat predict(cv::Mat values)
	{
		cv::Mat prediction = values * x;
		return prediction;
	};
	
	cv::Mat x; ///< The linear model we learn (\f$A*x = b\f$). TODO: Make private member variable

private:
	Regulariser regulariser; ///< Holding information about how to regularise.

	friend class boost::serialization::access;
	/**
	 * Serialises this class using boost::serialization.
	 *
	 * @param[in] ar The archive to serialise to (or to serialise from).
	 * @param[in] version An optional version argument.
	 */
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & x;
		ar & regulariser;
	}
};

} /* namespace superviseddescent */
#endif /* REGRESSORS_HPP_ */
