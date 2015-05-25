/*
 * superviseddescent: A C++11 implementation of the supervised descent
 *                    optimisation method
 * File: examples/rcr/model.hpp
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

#ifndef MODEL_HPP_
#define MODEL_HPP_

#include "helpers.hpp"

#include "opencv2/core/core.hpp"

#include <vector>
#include <string>

using cv::Mat;
using std::vector;
using std::string;

namespace rcr {

/**
 * Performs an initial alignment of the model, by putting the mean model into
 * the center of the face box.
 *
 * An optional scaling and translation parameters can be given to generate
 * perturbations of the initialisation.
 *
 * Note 02/04/15: I think with the new perturbation code, we can delete the optional
 * parameters here - make it as simple as possible, don't include what's not needed.
 * Align and perturb should really be separate - separate things.
 *
 * @param[in] mean Mean model points.
 * @param[in] faceBox A facebox to align the model to.
 * @param[in] scalingX Optional scaling in x of the model.
 * @param[in] scalingY Optional scaling in y of the model.
 * @param[in] translationX Optional translation in x of the model.
 * @param[in] translationY Optional translation in y of the model.
 * @return A cv::Mat of the aligned points.
 */
Mat alignMean(Mat mean, cv::Rect faceBox, float scalingX=1.0f, float scalingY=1.0f, float translationX=0.0f, float translationY=0.0f)
{
	// Initial estimate x_0: Center the mean face at the [-0.5, 0.5] x [-0.5, 0.5] square (assuming the face-box is that square)
	// More precise: Take the mean as it is (assume it is in a space [-0.5, 0.5] x [-0.5, 0.5]), and just place it in the face-box as
	// if the box is [-0.5, 0.5] x [-0.5, 0.5]. (i.e. the mean coordinates get upscaled)
	Mat alignedMean = mean.clone();
	Mat alignedMeanX = alignedMean.colRange(0, alignedMean.cols / 2);
	Mat alignedMeanY = alignedMean.colRange(alignedMean.cols / 2, alignedMean.cols);
	alignedMeanX = (alignedMeanX*scalingX + 0.5f + translationX) * faceBox.width + faceBox.x;
	alignedMeanY = (alignedMeanY*scalingY + 0.5f + translationY) * faceBox.height + faceBox.y;
	return alignedMean;
}

// Adaptive RCRC SDM update test
class InterEyeDistanceNormalisation
{
public:
	// params: The current landmark estimate
	inline cv::Mat operator()(cv::Mat params) {
		//getIed() // No access to the eye-identifiers here to normalise.
		vector<string> rightEyeIdentifiers{ "37", "40" };
		vector<string> leftEyeIdentifiers{ "43", "46" };
		vector<string> modelLandmarksList{ "37", "40", "43", "46", "49", "55", "58" };
		auto lmc = toLandmarkCollection(params, modelLandmarksList);
		auto ied = getIed(lmc, rightEyeIdentifiers, leftEyeIdentifiers);
		return Mat::ones(1, params.cols, params.type()) / ied;
	};
};

} /* namespace rcr */

#endif /* MODEL_HPP_ */
