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
#include "eos_core_landmark.hpp"

#include "cereal/cereal.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/vector.hpp"
#include "superviseddescent/matcerealisation.hpp"

#include "opencv2/core/core.hpp"

#include <vector>
#include <string>

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
cv::Mat alignMean(cv::Mat mean, cv::Rect faceBox, float scalingX=1.0f, float scalingY=1.0f, float translationX=0.0f, float translationY=0.0f)
{
	using cv::Mat;
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
	InterEyeDistanceNormalisation() = default;

	InterEyeDistanceNormalisation(std::vector<std::string> modelLandmarksList, std::vector<std::string> rightEyeIdentifiers, std::vector<std::string> leftEyeIdentifiers) : modelLandmarksList(modelLandmarksList), rightEyeIdentifiers(rightEyeIdentifiers), leftEyeIdentifiers(leftEyeIdentifiers)
	{
	};

	// params: The current landmark estimate
	inline cv::Mat operator()(cv::Mat params) {
		auto lmc = toLandmarkCollection(params, modelLandmarksList);
		auto ied = getIed(lmc, rightEyeIdentifiers, leftEyeIdentifiers);
		return cv::Mat::ones(1, params.cols, params.type()) / ied;
	};

private:
	std::vector<std::string> modelLandmarksList;
	std::vector<std::string> rightEyeIdentifiers;
	std::vector<std::string> leftEyeIdentifiers;

	friend class cereal::access;
	/**
	 * Serialises this class using cereal.
	 *
	 * @param[in] ar The archive to serialise to (or to serialise from).
	 */
	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(modelLandmarksList, rightEyeIdentifiers, leftEyeIdentifiers);
		//ar & BOOST_SERIALIZATION_NVP(modelLandmarksList) & BOOST_SERIALIZATION_NVP(rightEyeIdentifiers) & BOOST_SERIALIZATION_NVP(leftEyeIdentifiers);
	}
};

class detection_model
{
public:
	using model_type = superviseddescent::SupervisedDescentOptimiser<superviseddescent::LinearRegressor<PartialPivLUSolveSolverDebug>, InterEyeDistanceNormalisation>;
	detection_model() = default;

	detection_model(model_type optimised_model, cv::Mat mean, std::vector<std::string> landmark_ids, std::vector<rcr::HoGParam> hog_params, std::vector<std::string> right_eye_ids, std::vector<std::string> left_eye_ids) : optimised_model(optimised_model), mean(mean), landmark_ids(landmark_ids), hog_params(hog_params), right_eye_ids(right_eye_ids), left_eye_ids(left_eye_ids)
	{};

	eos::core::LandmarkCollection<cv::Vec2f> detect(cv::Mat image, cv::Rect facebox)
	{
		// Obtain the initial estimate using the mean landmarks:
		cv::Mat mean_initialisation = rcr::alignMean(mean, facebox);
		//rcr::drawLandmarks(image, mean_initialisation, { 0, 0, 255 });

		rcr::HogTransform hog({ image }, hog_params, landmark_ids, right_eye_ids, left_eye_ids);

		cv::Mat landmarks = optimised_model.predict(mean_initialisation, cv::Mat(), hog);

		return toLandmarkCollection(landmarks, landmark_ids);
	};

private:
	model_type optimised_model;
	//std::vector<cv::Mat> regressors; // Note: vector<LinearRegressorModel> or vector<LinearRegressor::model_type> instead?
	cv::Mat mean;						///< The mean of the model, learned and scaled from training data, given a specific face detector
	std::vector<rcr::HoGParam> hog_params;	///< The hog parameters for each regressor level
	std::vector<std::string> landmark_ids;			///< The landmark identifiers the model consists of
	std::vector<std::string> right_eye_ids, left_eye_ids;	///< Holds information about which landmarks are the eyes, to calculate the IED normalisation for the adaptive update

	friend class cereal::access;
	/**
	 * Serialises this class using cereal.
	 *
	 * @param[in] ar The archive to serialise to (or to serialise from).
	 */
	template<class Archive>
	void serialize(Archive& archive)
	{
		archive(optimised_model, mean, landmark_ids, hog_params, right_eye_ids, left_eye_ids);
	};

};

} /* namespace rcr */

#endif /* MODEL_HPP_ */
