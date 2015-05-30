/*
 * superviseddescent: A C++11 implementation of the supervised descent
 *                    optimisation method
 * File: superviseddescent/matcerealisation.hpp
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

#ifndef MATCEREALISATION_HPP_
#define MATCEREALISATION_HPP_

#include "opencv2/core/core.hpp"

#include "cereal/types/vector.hpp"

/**
 * Serialisation for the OpenCV cv::Mat class. Supports text and binary serialisation.
 *
 */

namespace cv {

/**
 * Serialize a cv::Mat using cereal.
 *
 * Supports all types of matrices as well as non-contiguous ones.
 *
 * @param[in] ar The archive to serialise to (or to serialise from).
 * @param[in] mat The matrix to serialise (or deserialise).
 */
template<class Archive>
void save(Archive& ar, const cv::Mat& mat)
{
	int rows, cols, type;
	bool continuous;

	// only when saving:
	rows = mat.rows;
	cols = mat.cols;
	type = mat.type();
	continuous = mat.isContinuous();
	//

	ar & rows & cols & type & continuous;

	if (continuous) {
		const int data_size = rows * cols * static_cast<int>(mat.elemSize());
		// This copies the data. I don't think there's a solution that avoids copying the data.
		std::vector<uchar> mat_data(mat.ptr(), mat.ptr() + data_size);
		ar & mat_data;
	}
	else {
		const int row_size = cols * static_cast<int>(mat.elemSize());
		std::vector<uchar> row_data(row_size);
		for (int i = 0; i < rows; i++) {
			row_data.assign(mat.ptr(i), mat.ptr(i) + row_size);
			ar & row_data;
		}
	}
};

template<class Archive>
void load(Archive& ar, cv::Mat& mat)
{
	int rows, cols, type;
	bool continuous;

	ar & rows & cols & type & continuous;

	// only when loading:
	//mat.create(1, 1, type); // create a pseudo matrix to extract the elemSize() later
	//

	if (continuous) {
		//const int data_size = rows * cols * static_cast<int>(mat.elemSize());
		// This copies the data. I don't think there's a solution that avoids copying the data.
		//std::vector<uchar> mat_data(mat.ptr(), mat.ptr() + data_size);
		std::vector<uchar> mat_data;
		ar & mat_data;
		mat = cv::Mat(rows, cols, type, &mat_data[0]).clone(); // without clone(), the data is destroyed as soon as the vector goes out of scope.
		// alternative: Mat(...).copyTo(mat)?
	}
	else {
		//const int row_size = cols * static_cast<int>(mat.elemSize());
		mat = cv::Mat(rows, cols, type);
		std::vector<uchar> row_data; // will hold the data read from the file for each row
		for (int i = 0; i < rows; i++) {
			ar & row_data;
			cv::Mat(1, cols, type, &row_data[0]).copyTo(mat.row(i));
			//row.assign(mat.ptr(i), mat.ptr(i) + row_size);
		}
	}
};

}

#endif /* MATCEREALISATION_HPP_ */
