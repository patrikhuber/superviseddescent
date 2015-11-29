/*
 * superviseddescent: A C++11 implementation of the supervised descent
 *                    optimisation method
 * File: rcr/model.hpp
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
#include "adaptive_vlhog.hpp"
#include "landmark.hpp"
#include "superviseddescent/superviseddescent.hpp"
#include "superviseddescent/verbose_solver.hpp"

#include "cereal/cereal.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/vector.hpp"
#include "cereal/archives/binary.hpp"
#include "superviseddescent/utils/mat_cerealisation.hpp"

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
 * @param[in] facebox A facebox to align the model to.
 * @param[in] scaling_x Optional scaling in x of the model.
 * @param[in] scaling_y Optional scaling in y of the model.
 * @param[in] translation_x Optional translation in x of the model.
 * @param[in] translation_y Optional translation in y of the model.
 * @return A cv::Mat of the aligned points.
 */
cv::Mat align_mean(cv::Mat mean, cv::Rect facebox, float scaling_x=1.0f, float scaling_y=1.0f, float translation_x=0.0f, float translation_y=0.0f)
{
	using cv::Mat;
	// Initial estimate x_0: Center the mean face at the [-0.5, 0.5] x [-0.5, 0.5] square (assuming the face-box is that square)
	// More precise: Take the mean as it is (assume it is in a space [-0.5, 0.5] x [-0.5, 0.5]), and just place it in the face-box as
	// if the box is [-0.5, 0.5] x [-0.5, 0.5]. (i.e. the mean coordinates get upscaled)
	Mat aligned_mean = mean.clone();
	Mat aligned_mean_x = aligned_mean.colRange(0, aligned_mean.cols / 2);
	Mat aligned_mean_y = aligned_mean.colRange(aligned_mean.cols / 2, aligned_mean.cols);
	aligned_mean_x = (aligned_mean_x*scaling_x + 0.5f + translation_x) * facebox.width + facebox.x;
	aligned_mean_y = (aligned_mean_y*scaling_y + 0.5f + translation_y) * facebox.height + facebox.y;
	return aligned_mean;
}

/**
 * This class handles the adaptive shape update of the RCR
 * during training. The update is normalised by calculating the
 * IED of the given \c params and then returning a normalisation
 * vector 1/IED.
 */
class InterEyeDistanceNormalisation
{
public:
	InterEyeDistanceNormalisation() = default;

	InterEyeDistanceNormalisation(std::vector<std::string> modelLandmarksList, std::vector<std::string> rightEyeIdentifiers, std::vector<std::string> leftEyeIdentifiers) : modelLandmarksList(modelLandmarksList), rightEyeIdentifiers(rightEyeIdentifiers), leftEyeIdentifiers(leftEyeIdentifiers)
	{
	};

	// params: The current landmark estimate
	inline cv::Mat operator()(cv::Mat params) {
		auto lmc = to_landmark_collection(params, modelLandmarksList);
		auto ied = get_ied(lmc, rightEyeIdentifiers, leftEyeIdentifiers);
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
	}
};

/**
 * This class represents a learned RCR landmark detection
 * model. It can detect landmarks and can be stored/loaded.
 */
class detection_model
{
public:
	using model_type = superviseddescent::SupervisedDescentOptimiser<superviseddescent::LinearRegressor<superviseddescent::VerbosePartialPivLUSolver>, InterEyeDistanceNormalisation>;
	detection_model() = default;

	detection_model(model_type optimised_model, cv::Mat mean, std::vector<std::string> landmark_ids, std::vector<rcr::HoGParam> hog_params, std::vector<std::string> right_eye_ids, std::vector<std::string> left_eye_ids) : optimised_model(optimised_model), mean(mean), landmark_ids(landmark_ids), hog_params(hog_params), right_eye_ids(right_eye_ids), left_eye_ids(left_eye_ids)
	{};

	// Run the model from a fbox. i.e. init using the mean, then optimise.
	LandmarkCollection<cv::Vec2f> detect(cv::Mat image, cv::Rect facebox)
	{
		// Obtain the initial estimate using the mean landmarks:
		cv::Mat mean_initialisation = rcr::align_mean(mean, facebox);
		//rcr::draw_landmarks(image, mean_initialisation, { 0, 0, 255 });

		std::vector<cv::Mat> images{ image }; // we can't pass a temporary anymore to HogTransform
		rcr::HogTransform hog(images, hog_params, landmark_ids, right_eye_ids, left_eye_ids);

		cv::Mat landmarks = optimised_model.predict(mean_initialisation, cv::Mat(), hog);

		return to_landmark_collection(landmarks, landmark_ids);
	};

	// Run the model from landmark init, e.g. using the previous frame's lms. I.e. do not init from the mean. Directly optimise.
	LandmarkCollection<cv::Vec2f> detect(cv::Mat image, cv::Mat initialisation)
	{
		//rcr::drawLandmarks(image, initialisation, { 0, 0, 255 });

		std::vector<cv::Mat> images{ image };
		rcr::HogTransform hog(images, hog_params, landmark_ids, right_eye_ids, left_eye_ids);

		cv::Mat landmarks = optimised_model.predict(initialisation, cv::Mat(), hog);

		return to_landmark_collection(landmarks, landmark_ids);
	};

	cv::Mat get_mean()
	{
		return mean;
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

/**
 * Load a trained \c detection_model that was stored
 * as cereal::BinaryInputArchive from the harddisk.
 *
 * @param[in] filename Filename to a model.
 * @return The loaded detection_model.
 */
detection_model load_detection_model(std::string filename)
{
	detection_model rcr_model;
	
	std::ifstream file(filename, std::ios::binary);
	cereal::BinaryInputArchive input_archive(file);
	input_archive(rcr_model);

	return rcr_model;
};

/**
 * Save a trained \c detection_model to the harddisk
 * as cereal::BinaryInputArchive.
 *
 * @param[in] model The model to be saved.
 * @param[in] filename Filename for the model.
 */
void save_detection_model(detection_model model, std::string filename)
{
	std::ofstream file(filename, std::ios::binary);
	cereal::BinaryOutputArchive output_archive(file);
	output_archive(model);
};

} /* namespace rcr */

#endif /* MODEL_HPP_ */
