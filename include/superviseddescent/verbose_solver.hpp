/*
 * superviseddescent: A C++11 implementation of the supervised descent
 *                    optimisation method
 * File: superviseddescent/verbose_solver.hpp
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

#ifndef VERBOSE_SOLVER_HPP_
#define VERBOSE_SOLVER_HPP_

#include "superviseddescent/regressors.hpp"

#include "Eigen/LU"

#include "opencv2/core/core.hpp"

#include <iostream>
#include <chrono>

namespace superviseddescent {

/**
 * A solver that the LinearRegressor uses to solve its system of linear
 * equations. This class is exactly the same as \c PartialPivLUSolver, with
 * the difference that this solver prints out information about timing and
 * the exact regularisation parameter. It is helpful for debugging purposes.
 * It has no runtime overhead, but it prints quite some information to
 * the standard output.
 */
class VerbosePartialPivLUSolver
{
public:
	/**
	 * The same as \c PartialPivLUSolver, with the difference that this solver
	 * prints out information about timing and the exact regularisation parameter.
	 *
	 * @copydoc PartialPivLUSolver::solve(cv::Mat data, cv::Mat labels, superviseddescent::Regulariser regulariser).
	 */
	cv::Mat solve(cv::Mat data, cv::Mat labels, superviseddescent::Regulariser regulariser)
	{
		using cv::Mat;
		using std::cout;
		using std::endl;
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
		Mat regularisationMatrix = regulariser.get_matrix(AtA_Map, data.rows);
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

} /* namespace superviseddescent */
#endif /* VERBOSE_SOLVER_HPP_ */
