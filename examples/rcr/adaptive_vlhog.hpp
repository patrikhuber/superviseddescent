/*
 * superviseddescent: A C++11 implementation of the supervised descent
 *                    optimisation method
 * File: examples/rcr/adaptive_vlhog.hpp
 *
 * Copyright 2015 Patrik Huber
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

#ifndef ADAPTIVE_VLHOG_HPP_
#define ADAPTIVE_VLHOG_HPP_

extern "C" {
	#include "../hog.h" // From the VLFeat C library
}

#include "helpers.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <vector>
#include <string>

namespace rcr {

struct HoGParam
{
	VlHogVariant vlHogVariant;
	int numCells; int cellSize; int numBins; int resizeTo;
	// note: alternatively, we could dynamically vary cellSize. Guess it works if the hog features are somehow normalised.

private:
	friend class cereal::access;
	/**
	 * Serialises this class using cereal.
	 *
	 * @param[in] ar The archive to serialise to (or to serialise from).
	 */
	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(vlHogVariant, numCells, cellSize, numBins, resizeTo);
	}
};

/**
 * Function object that extracts HoG features at given 2D landmark locations
 * and returns them as a row vector.
 *
 * We wrap all the C-style memory allocations of the VLFeat library
 * in cv::Mat's.
 * Note: Any other library and features can of course be used.
 */
class HogTransform
{
public:
	/**
	 * Constructs a HoG transform with given images and parameters.
	 * The images are needed so the HoG features can be extracted from the images
	 * when the SupervisedDescentOptimiser calls this functor (the optimiser
	 * itself doesn't know and care about the images).
	 *
	 * Note: \p images can consist of only 1 image, when using the model for
	 * prediction on new images.
	 *
	 * Note: Only VlHogVariantUoctti is tested.
	 *
	 * @param[in] images A vector of images used for training or testing.
	 * @param[in] vlHogVariant The VLFeat HoG variant.
	 * @param[in] numCells Number of HoG cells that should be constructed around each landmark.
	 * @param[in] cellSize Width of one HoG cell in pixels.
	 * @param[in] numBins Number of orientations of a HoG cell.
	 */
	HogTransform(std::vector<cv::Mat> images, std::vector<HoGParam> hog_params, std::vector<std::string> modelLandmarksList, std::vector<std::string> rightEyeIdentifiers, std::vector<std::string> leftEyeIdentifiers) : images(images), hog_params(hog_params), modelLandmarksList(modelLandmarksList), rightEyeIdentifiers(rightEyeIdentifiers), leftEyeIdentifiers(leftEyeIdentifiers)
	{
	};

	/**
	 * Uses the current parameters (the 2D landmark locations, in SDM
	 * terminology the \c x), and extracts HoG features at these positions.
	 * These HoG features are the new \c y values.
	 *
	 * The 2D landmark position estimates are given as a row vector
	 * [x_0, ..., x_n, y_0, ..., y_n].
	 *
	 * @param[in] parameters The current 2D landmark position estimate.
	 * @param[in] regressorLevel Not used in this example.
	 * @param[in] trainingIndex Gets supplied by the SupervisedDescent optimiser during training and testing, to know from which image to extract features.
	 * @return Returns the HoG features at the given 2D locations.
	 */
	cv::Mat operator()(cv::Mat parameters, size_t regressorLevel, int trainingIndex = 0)
	{
		assert(parameters.rows == 1);
		using cv::Mat;
			
		Mat grayImage;
		if (images[trainingIndex].channels() == 3) {
			cv::cvtColor(images[trainingIndex], grayImage, cv::COLOR_BGR2GRAY);
		}
		else {
			grayImage = images[trainingIndex];
		}

		//int patchWidthHalf = hog_params[regressorLevel].numCells * (hog_params[regressorLevel].cellSize / 2);
		int patchWidthHalf = std::round(get_ied(to_landmark_collection(parameters, modelLandmarksList), rightEyeIdentifiers, leftEyeIdentifiers) / 3); // Could use the formula of Zhenhua here. Well I used it somewhere in the old code?

		Mat hogDescriptors; // We'll get the dimensions later from vl_hog_get_*

		int numLandmarks = parameters.cols / 2;
		for (int i = 0; i < numLandmarks; ++i) {
			int x = cvRound(parameters.at<float>(i));
			int y = cvRound(parameters.at<float>(i + numLandmarks));

			Mat roiImg;
			if (x - patchWidthHalf < 0 || y - patchWidthHalf < 0 || x + patchWidthHalf >= grayImage.cols || y + patchWidthHalf >= grayImage.rows) {
				// The feature extraction location is too far near a border. We extend the
				// image (add a black canvas) and then extract from this larger image.
				int borderLeft = (x - patchWidthHalf) < 0 ? std::abs(x - patchWidthHalf) : 0; // x and y are patch-centers
				int borderTop = (y - patchWidthHalf) < 0 ? std::abs(y - patchWidthHalf) : 0;
				int borderRight = (x + patchWidthHalf) >= grayImage.cols ? std::abs(grayImage.cols - (x + patchWidthHalf)) : 0;
				int borderBottom = (y + patchWidthHalf) >= grayImage.rows ? std::abs(grayImage.rows - (y + patchWidthHalf)) : 0;
				Mat extendedImage = grayImage.clone();
				cv::copyMakeBorder(extendedImage, extendedImage, borderTop, borderBottom, borderLeft, borderRight, cv::BORDER_CONSTANT, cv::Scalar(0));
				cv::Rect roi((x - patchWidthHalf) + borderLeft, (y - patchWidthHalf) + borderTop, patchWidthHalf * 2, patchWidthHalf * 2); // Rect: x y w h. x and y are top-left corner.
				roiImg = extendedImage(roi).clone(); // clone because we need a continuous memory block
			}
			else {
				cv::Rect roi(x - patchWidthHalf, y - patchWidthHalf, patchWidthHalf * 2, patchWidthHalf * 2); // x y w h. Rect: x and y are top-left corner. Our x and y are center. Convert.
				roiImg = grayImage(roi).clone(); // clone because we need a continuous memory block
			}

			// new:
			cv::resize(roiImg, roiImg, { hog_params[regressorLevel].resizeTo, hog_params[regressorLevel].resizeTo });

			roiImg.convertTo(roiImg, CV_32FC1); // vl_hog_put_image expects a float* (values 0.0f-255.0f)
			VlHog* hog = vl_hog_new(hog_params[regressorLevel].vlHogVariant, hog_params[regressorLevel].numBins, false); // transposed (=col-major) = false
			vl_hog_put_image(hog, (float*)roiImg.data, roiImg.cols, roiImg.rows, 1, hog_params[regressorLevel].cellSize); // (the '1' is numChannels)
			int ww = static_cast<int>(vl_hog_get_width(hog)); // assert ww == hh == numCells
			int hh = static_cast<int>(vl_hog_get_height(hog));
			int dd = static_cast<int>(vl_hog_get_dimension(hog)); // assert ww=hogDim1, hh=hogDim2, dd=hogDim3
			Mat hogArray(1, ww*hh*dd, CV_32FC1); // safer & same result. Don't use C-style memory management.
			vl_hog_extract(hog, hogArray.ptr<float>(0));
			vl_hog_delete(hog);
			Mat hogDescriptor(hh*ww*dd, 1, CV_32FC1);
			// Stack the third dimensions of the HOG descriptor of this patch one after each other in a column-vector:
			for (int j = 0; j < dd; ++j) {
				Mat hogFeatures(hh, ww, CV_32FC1, hogArray.ptr<float>(0) + j*ww*hh); // Creates the same array as in Matlab. I might have to check this again if hh!=ww (non-square)
				hogFeatures = hogFeatures.t(); // necessary because the Matlab reshape() takes column-wise from the matrix while the OpenCV reshape() takes row-wise.
				hogFeatures = hogFeatures.reshape(0, hh*ww); // make it to a column-vector
				Mat currentDimSubMat = hogDescriptor.rowRange(j*ww*hh, j*ww*hh + ww*hh);
				hogFeatures.copyTo(currentDimSubMat);
			}
			hogDescriptor = hogDescriptor.t(); // now a row-vector
			hogDescriptors.push_back(hogDescriptor);
		}
		// concatenate all the descriptors for this sample vertically (into a row-vector):
		hogDescriptors = hogDescriptors.reshape(0, hogDescriptors.cols * numLandmarks).t();

		// add a bias row (affine part)
		Mat bias = Mat::ones(1, 1, CV_32FC1);
		cv::hconcat(hogDescriptors, bias, hogDescriptors);
		return hogDescriptors;
	};

private:
	std::vector<cv::Mat> images;
	std::vector<HoGParam> hog_params;

	// For the IED normalisation / adaptive part:
	std::vector<std::string> modelLandmarksList;
	std::vector<std::string> rightEyeIdentifiers;
	std::vector<std::string> leftEyeIdentifiers;
};


} /* namespace rcr */

#endif /* ADAPTIVE_VLHOG_HPP_ */
