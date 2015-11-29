/*
 * superviseddescent: A C++11 implementation of the supervised descent
 *                    optimisation method
 * File: rcr/adaptive_vlhog.hpp
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
	#include "hog.h" // From the VLFeat C library
}

#include "helpers.hpp"

#include "cereal/access.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <vector>
#include <string>

namespace rcr {

struct HoGParam
{
	VlHogVariant vlhog_variant;
	int num_cells; int cell_size; int num_bins;
	float relative_patch_size; // the patch size we'd like in percent of the IED of the current image
	// note: alternatively, we could dynamically vary cell_size. Guess it works if the hog features are somehow normalised.

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
		ar(vlhog_variant, num_cells, cell_size, num_bins, relative_patch_size);
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
	 * Do not call with \p images that are temporaries!
	 *
	 * Note: Only VlHogVariantUoctti is tested.
	 *
	 * @param[in] images A vector of images used for training or testing.
	 * @param[in] hog_params Parameters for the VLFeat HOG features.
	 * @param[in] modelLandmarksList Todo.
	 * @param[in] rightEyeIdentifiers Todo.
	 * @param[in] leftEyeIdentifiers Todo.
	 */
	HogTransform(const std::vector<cv::Mat>& images, std::vector<HoGParam> hog_params, std::vector<std::string> modelLandmarksList, std::vector<std::string> rightEyeIdentifiers, std::vector<std::string> leftEyeIdentifiers) : images(images), hog_params(hog_params), modelLandmarksList(modelLandmarksList), rightEyeIdentifiers(rightEyeIdentifiers), leftEyeIdentifiers(leftEyeIdentifiers)
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

		// This is in pixels in the original image:
		int patch_width_half = std::round(hog_params[regressorLevel].relative_patch_size * get_ied(to_landmark_collection(parameters, modelLandmarksList), rightEyeIdentifiers, leftEyeIdentifiers) / 2); // Could use the formula of Zhenhua here, but we can use it in the app if desired

		// Ideally, these two values should be similar. If the second is much bigger, it's probably of not much use (we just upscale the images for nothing)
		//std::cout << "pw: " << patch_width_half * 2 << ", rs: " << hog_params[regressorLevel].num_cells * hog_params[regressorLevel].cell_size << std::endl;

		Mat hogDescriptors; // We'll get the dimensions later from vl_hog_get_*

		int numLandmarks = parameters.cols / 2;
		for (int i = 0; i < numLandmarks; ++i) {
			int x = cvRound(parameters.at<float>(i));
			int y = cvRound(parameters.at<float>(i + numLandmarks));

			Mat roiImg;
			if (x - patch_width_half < 0 || y - patch_width_half < 0 || x + patch_width_half >= grayImage.cols || y + patch_width_half >= grayImage.rows) {
				// The feature extraction location is too far near a border. We extend the
				// image (add a black canvas) and then extract from this larger image.
				int borderLeft = (x - patch_width_half) < 0 ? std::abs(x - patch_width_half) : 0; // x and y are patch-centers
				int borderTop = (y - patch_width_half) < 0 ? std::abs(y - patch_width_half) : 0;
				int borderRight = (x + patch_width_half) >= grayImage.cols ? std::abs(grayImage.cols - (x + patch_width_half)) : 0;
				int borderBottom = (y + patch_width_half) >= grayImage.rows ? std::abs(grayImage.rows - (y + patch_width_half)) : 0;
				Mat extendedImage = grayImage.clone();
				cv::copyMakeBorder(extendedImage, extendedImage, borderTop, borderBottom, borderLeft, borderRight, cv::BORDER_CONSTANT, cv::Scalar(0));
				cv::Rect roi((x - patch_width_half) + borderLeft, (y - patch_width_half) + borderTop, patch_width_half * 2, patch_width_half * 2); // Rect: x y w h. x and y are top-left corner.
				roiImg = extendedImage(roi).clone(); // clone because we need a continuous memory block
			}
			else {
				cv::Rect roi(x - patch_width_half, y - patch_width_half, patch_width_half * 2, patch_width_half * 2); // x y w h. Rect: x and y are top-left corner. Our x and y are center. Convert.
				roiImg = grayImage(roi).clone(); // clone because we need a continuous memory block
			}

			// This has to be the same for each image, so each image's HOG descriptor will have the same dimensions, independent of the image's resolution
			int fixed_roi_size = hog_params[regressorLevel].num_cells * hog_params[regressorLevel].cell_size;
			cv::resize(roiImg, roiImg, { fixed_roi_size, fixed_roi_size });

			roiImg.convertTo(roiImg, CV_32FC1); // vl_hog_put_image expects a float* (values 0.0f-255.0f)
			VlHog* hog = vl_hog_new(hog_params[regressorLevel].vlhog_variant, hog_params[regressorLevel].num_bins, false); // transposed (=col-major) = false
			vl_hog_put_image(hog, (float*)roiImg.data, roiImg.cols, roiImg.rows, 1, hog_params[regressorLevel].cell_size); // (the '1' is numChannels)
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
	const std::vector<cv::Mat>& images; // We store a reference so that each thread that gets spawned doesn't create a copy of this vector. An alternative might be to rethink the design and not store all images here.
	std::vector<HoGParam> hog_params;

	// For the IED normalisation / adaptive part:
	std::vector<std::string> modelLandmarksList;
	std::vector<std::string> rightEyeIdentifiers;
	std::vector<std::string> leftEyeIdentifiers;
};


} /* namespace rcr */

#endif /* ADAPTIVE_VLHOG_HPP_ */
