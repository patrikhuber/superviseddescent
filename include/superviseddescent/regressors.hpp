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

#include "cereal/cereal.hpp"
#include "superviseddescent/utils/mat_cerealisation.hpp"

#include "Eigen/Dense"

#include "opencv2/core/core.hpp"

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
	 * \c regularise_last_row is useful to specify the regularisation behaviour in
	 * the case the last row of the data matrix contains an affine (offset or
	 * bias) component. In that case, you might not want to regularise it (or
	 * maybe you do).
	 *
	 * @param[in] regularisation_type Specifies how to calculate lambda.
	 * @param[in] param Lambda, or a factor, depending on regularisationType.
	 * @param[in] regularise_last_row Specifies if the last row should be regularised.
	 */
	Regulariser(RegularisationType regularisation_type = RegularisationType::Manual, float param = 0.0f, bool regularise_last_row = true) : regularisation_type(regularisation_type), lambda(param), regularise_last_row(regularise_last_row)
	{
	};

	/**
	 * Calculates a diagonal regularisation matrix, given the RegularisationType
	 * and param of the regulariser. It will have the same dimensions as the
	 * given data matrix.
	 *
	 * @param[in] data Data matrix that might be used to calculate an automatic value for lambda.
	 * @param[in] num_training_elements Number of training elements.
	 * @return Returns a diagonal regularisation matrix with the same dimensions as the given data matrix.
	 */
	cv::Mat get_matrix(cv::Mat data, int num_training_elements)
	{
		switch (regularisation_type)
		{
		case RegularisationType::Manual:
			// We just take lambda as it was given, no calculation necessary.
			break;
		case RegularisationType::MatrixNorm:
			// The given lambda is the factor we have to multiply the automatic value with:
			lambda = lambda * static_cast<float>(cv::norm(data)) / static_cast<float>(num_training_elements);
			break;
		default:
			break;
		}

		cv::Mat regulariser = cv::Mat::eye(data.rows, data.cols, CV_32FC1) * lambda;

		if (!regularise_last_row) {
			// no lambda for the bias:
			regulariser.at<float>(regulariser.rows - 1, regulariser.cols - 1) = 0.0f;
		}
		return regulariser;
	};

private:
	RegularisationType regularisation_type; ///< The type of regularisation this regulariser is using.
	float lambda; ///< The parameter for RegularisationType. Can be lambda directly or a factor with which the lambda from MatrixNorm will be multiplied with.
	bool regularise_last_row; ///< If the last row of data matrix is a bias (offset), then you might want to choose whether it should be regularised as well. Otherwise, just leave it to default (true).

	friend class cereal::access;
	/**
	 * Serialises this class using cereal.
	 *
	 * Note: If we split the optimisation and the model, we should be able to
	 * delete this. There shouldn't be a need to serialise the regulariser!
	 *
	 * @param[in] ar The archive to serialise to (or to serialise from).
	 */
	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(regularisation_type, lambda, regularise_last_row);
	}
};

/**
 * A solver that the LinearRegressor uses to solve its system of linear
 * equations. It needs a solve function with the following signature:
 * \c cv::Mat solve(cv::Mat data, cv::Mat labels, Regulariser regulariser)
 *
 * The \c PartialPivLUSolver is a fast solver but it can't check for invertibility.
 * It supports parallel solving if compiled with OpenMP enabled.
 * Uses PartialPivLU::solve() instead of inverting the matrix.
 */
class PartialPivLUSolver
{
public:
	/**
	 * Solves the linear system \f$ (\text{data}^\text{T} * \text{data} + \text{regulariser}) * \text{X} = \text{data}^\text{T} * \text{labels}\f$
	 * where regulariser is a diagonal matrix. This results in a least-squares
	 * approximation of the original system.
	 * \c labels can consist of multiple columns.
	 *
	 * Note/Todo: we should leave the choice of inverting A or AtA to the solver.
	 *  But this also means we need to pass through the regularisation params.
	 *  We can't just pass a cv::Mat regularisation because the dimensions for
	 *  regularising A and AtA are different.
	 *
	 * @param[in] data Data matrix with each row being a data sample.
	 * @param[in] labels Labels for each data sample.
	 * @param[in] regulariser A regularisation.
	 * @return The solution matrix.
	 */
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
		Mat regularisation_matrix = regulariser.get_matrix(AtA_Map, data.rows);
		Eigen::Map<RowMajorMatrixXf> reg_Eigen(regularisation_matrix.ptr<float>(), regularisation_matrix.rows, regularisation_matrix.cols);

		Eigen::DiagonalMatrix<float, Eigen::Dynamic> reg_Eigen_diag(regularisation_matrix.rows);
		Eigen::VectorXf diag_vec(regularisation_matrix.rows);
		for (int i = 0; i < diag_vec.size(); ++i) {
			diag_vec(i) = regularisation_matrix.at<float>(i, i);
		}
		reg_Eigen_diag.diagonal() = diag_vec;
		AtA_Eigen = AtA_Eigen + reg_Eigen_diag.toDenseMatrix();

		// Perform a fast PartialPivLU and use ::solve() (better than inverting):
		Eigen::PartialPivLU<RowMajorMatrixXf> lu_of_AtA(AtA_Eigen);
		RowMajorMatrixXf x_Eigen = lu_of_AtA.solve(A_Eigen.transpose() * labels_Eigen);
		//RowMajorMatrixXf x_Eigen = AtA_Eigen.partialPivLu.solve(A_Eigen.transpose() * labels_Eigen);

		// Map the resulting x back to a cv::Mat by creating a Mat header:
		Mat x(static_cast<int>(x_Eigen.rows()), static_cast<int>(x_Eigen.cols()), CV_32FC1, x_Eigen.data());

		// We have to copy the data because the underlying data is managed by Eigen::Matrix x_Eigen, which will go out of scope after we leave this function:
		return x.clone();
		//return qrOfAtA.isInvertible();
	};
};

/**
 * A solver that the LinearRegressor uses to solve its system of linear
 * equations. It needs a solve function with the following signature:
 * \c cv::Mat solve(cv::Mat data, cv::Mat labels, Regulariser regulariser)
 *
 * The \c ColPivHouseholderQRSolver can check for invertibility, but it is much
 * MUCH slower than a \c PartialPivLUSolver.
 */
class ColPivHouseholderQRSolver
{
public:
	/**
	 * Solves the linear system \f$ (\text{data}^t * \text{data} + \text{regulariser}) * X = \text{data}^t * \text{labels}\f$
	 * where \c Regulariser is a diagonal matrix. This results in a least-squares
	 * approximation of the original system.
	 * \c labels can consist of multiple columns.
	 *
	 * Note/Todo: we should leave the choice of inverting A or AtA to the solver.
	 *  But this also means we need to pass through the regularisation params.
	 *  We can't just pass a cv::Mat regularisation because the dimensions for
	 *  regularising A and AtA are different.
	 *
	 * @param[in] data Data matrix with each row being a data sample.
	 * @param[in] labels Labels for each data sample.
	 * @param[in] regulariser A regularisation.
	 * @return The solution matrix.
	 */
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
		Mat regularisation_matrix = regulariser.get_matrix(AtA_Map, data.rows);
		Eigen::Map<RowMajorMatrixXf> reg_Eigen(regularisation_matrix.ptr<float>(), regularisation_matrix.rows, regularisation_matrix.cols);

		Eigen::DiagonalMatrix<float, Eigen::Dynamic> reg_Eigen_diag(regularisation_matrix.rows);
		Eigen::VectorXf diag_vec(regularisation_matrix.rows);
		for (int i = 0; i < diag_vec.size(); ++i) {
			diag_vec(i) = regularisation_matrix.at<float>(i, i);
		}
		reg_Eigen_diag.diagonal() = diag_vec;
		AtA_Eigen = AtA_Eigen + reg_Eigen_diag.toDenseMatrix();

		// Perform a ColPivHouseholderQR (faster than FullPivLU) that allows to check for invertibility:
		Eigen::ColPivHouseholderQR<RowMajorMatrixXf> qr_of_AtA(AtA_Eigen);
		auto rankOfAtA = qr_of_AtA.rank();
		if (!qr_of_AtA.isInvertible()) {
			// Eigen may return garbage (their docu is not very specific). Best option is to increase regularisation.
			std::cout << "The regularised AtA is not invertible. We continued learning, but Eigen may return garbage (their docu is not very specific). (The rank is " << std::to_string(rankOfAtA) << ", full rank would be " << std::to_string(AtA_Eigen.rows()) << "). Increase lambda." << std::endl;
		}
		RowMajorMatrixXf AtAInv_Eigen = qr_of_AtA.inverse();

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
template<class Solver = PartialPivLUSolver>
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

	friend class cereal::access;
	/**
	 * Serialises this class using cereal.
	 *
	 * @param[in] ar The archive to serialise to (or to serialise from).
	 */
	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(x, regulariser);
	}
};

} /* namespace superviseddescent */
#endif /* REGRESSORS_HPP_ */
