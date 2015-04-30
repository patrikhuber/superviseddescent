/*
 * superviseddescent: A C++11 implementation of the supervised descent
 *                    optimisation method
 * File: superviseddescent/regressors.hpp
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

#ifndef REGRESSORS_HPP_
#define REGRESSORS_HPP_

#include "superviseddescent/matserialisation.hpp"

#include "opencv2/core/core.hpp"
#include "Eigen/Dense"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/serialization/serialization.hpp"

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
	void serialize(Archive& ar, const unsigned int version)
	{
		ar & regularisationType;
		ar & lambda;
		ar & regulariseLastRow;
	}
};


/**
 * A solver that the LinearRegressor uses to solve its system of linear
 * equations. It needs a solve function with the following signature:
 * \c cv::Mat solve(cv::Mat data, cv::Mat labels, Regulariser regulariser)
 *
 * The \c ColPivHouseholderQRSolver can check for invertibility, but it is much
 * slower than a \c PartialPivLUSolver.
 */
class ColPivHouseholderQRSolver
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
		// Map the cv::Mat data and labels to Eigen matrices:
		Eigen::Map<RowMajorMatrixXf> A_Eigen(data.ptr<float>(), data.rows, data.cols);
		Eigen::Map<RowMajorMatrixXf> labels_Eigen(labels.ptr<float>(), labels.rows, labels.cols);

		RowMajorMatrixXf AtA_Eigen = A_Eigen.transpose() * A_Eigen;

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
		AtA_Eigen = AtA_Eigen + reg_Eigen_diag.toDenseMatrix();

		// Perform a ColPivHouseholderQR (faster than FullPivLU) that allows to check for invertibility:
		Eigen::ColPivHouseholderQR<RowMajorMatrixXf> qrOfAtA(AtA_Eigen);
		auto rankOfAtA = qrOfAtA.rank();
		if (!qrOfAtA.isInvertible()) {
			// Eigen may return garbage (their docu is not very specific). Best option is to increase regularisation.
			std::cout << "The regularised AtA is not invertible. We continued learning, but Eigen may return garbage (their docu is not very specific). (The rank is " << std::to_string(rankOfAtA) << ", full rank would be " << std::to_string(AtA_Eigen.rows()) << "). Increase lambda." << std::endl;
		}
		RowMajorMatrixXf AtAInv_Eigen = qrOfAtA.inverse();

		// x = (AtAReg)^-1 * At * b:
		RowMajorMatrixXf x_Eigen = AtAInv_Eigen * A_Eigen.transpose() * labels_Eigen;

		// Map the resulting x back to a cv::Mat by creating a Mat header:
		Mat x(static_cast<int>(x_Eigen.rows()), static_cast<int>(x_Eigen.cols()), CV_32FC1, x_Eigen.data());

		// We have to copy the data because the underlying data is managed by Eigen::Matrix x_Eigen, which will go out of scope after we leave this function:
		return x.clone();
		//return qrOfAtA.isInvertible();
	};
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
template<class Solver = ColPivHouseholderQRSolver>
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
	 * Note/Todo: We probably want to change the interface to return void. Not
	 * all solvers can return a bool, it's kind of optional, so we can't rely on it.
	 *
	 * @param[in] data Training data matrix, one example per row.
	 * @param[in] labels Labels corresponding to the training data.
	 * @return Returns whether \f$ \text{data}^t * \text{data} \f$ was invertible. (Note: Always returns true at the moment.)
	 */
	bool learn(cv::Mat data, cv::Mat labels) override
	{
		cv::Mat x = solver.solve(data, labels, regulariser);
		this->x = x;
		return true; // see todo above
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
	double test(cv::Mat data, cv::Mat labels) override
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
	cv::Mat predict(cv::Mat values) override
	{
		cv::Mat prediction = values * x;
		return prediction;
	};
	
	cv::Mat x; ///< The linear model we learn (\f$A*x = b\f$). TODO: Make private member variable

private:
	Regulariser regulariser; ///< Holding information about how to regularise.
	Solver solver; ///< The type of solver used to solve the regressors linear system of equations.

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
		ar & x;
		ar & regulariser;
	}
};

} /* namespace superviseddescent */
#endif /* REGRESSORS_HPP_ */
