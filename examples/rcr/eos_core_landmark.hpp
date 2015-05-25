/*
 * superviseddescent: A C++11 implementation of the supervised descent
 *                    optimisation method
 * File: examples/rcr/eos_core_landmark.hpp
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

#ifndef LANDMARKS_HPP_
#define LANDMARKS_HPP_

#include <string>
#include <vector>

namespace eos {
	namespace core {

template<class LandmarkType>
struct Landmark
{
	std::string name;
	LandmarkType coordinates;
};

template<class LandmarkType> using LandmarkCollection = std::vector<Landmark<LandmarkType>>;

// I guess ranges would make this a lot easier.
template<class T>
LandmarkCollection<T> filter(const LandmarkCollection<T>& landmarks, const std::vector<std::string>& filter)
{
	LandmarkCollection<T> filteredLandmarks;
	using std::begin;
	using std::end;
	std::copy_if(begin(landmarks), end(landmarks), std::back_inserter(filteredLandmarks),
		[&](const Landmark<T>& lm) { return std::find(begin(filter), end(filter), lm.name) != end(filter); }
	);
	return filteredLandmarks;
};

	} /* namespace core */
} /* namespace eos */

#endif /* LANDMARKS_HPP_ */
