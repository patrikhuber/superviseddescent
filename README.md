# superviseddescent: A C++11 implementation of the supervised descent optimisation method
[![GitHub release](http://img.shields.io/github/release/patrikhuber/superviseddescent.svg?style=flat-square)][release]
[![Apache License 2.0](https://img.shields.io/github/license/patrikhuber/superviseddescent.svg?style=flat-square)][license]

[release]: https://github.com/patrikhuber/superviseddescent/releases
[license]: https://github.com/patrikhuber/superviseddescent/blob/master/LICENSE

superviseddescent is a C++11 implementation of the supervised descent method, which is a generic algorithm to perform optimisation of arbitrary functions.

The library contains an implementation of the _Robust Cascaded Regression_ facial landmark detection and features a pre-trained detection model.

There are two main advantages compared to traditional optimisation algorithms like gradient descent, L-BFGS and the like:
* The function doesn't have to be differentiable. It works with arbitrary functions, for example with the HoG operator (which is a non-differentiable function)
* It might be better at reaching a global optimum - give it a try!

The theory is based on the idea of _Supervised Descent Method and Its Applications to Face Alignment_, from X. Xiong & F. De la Torre, CVPR 2013 (http://ieeexplore.ieee.org/xpl/articleDetails.jsp?tp=&arnumber=6618919)

## Features

* Generic implementation, can be used to optimise arbitrary parameters and functions
* Fast, using Eigen for matrix operations
* Modern, clean C++11/14
* Header only (compilation/installation only for the examples and tests)
* Fully cross-platform compatible - learned models can be run even on phones

## Usage

It is a header only library, and can thus be included directly into your project by just adding `superviseddescent/include` and `3rdparty/cereal-1.1.1/include` to your project's include directory.

* Tested with the following compilers: gcc-4.8.2, clang-3.5, Visual Studio 2013 & 2015 RC
* Needed dependencies: OpenCV core (2.4.3), Eigen (3.2). For the apps, Boost (1.54.0) is also required. Older versions of the libraries might work as well.

## Sample code

### Robust Cascaded Regression landmark detection:

While the method and this library can be used to approximate arbitrary functions, its prime example is facial landmark detection. The library contains an implementation of the _Robust Cascaded Regression_ landmark detection and a pre-trained model. Running it is as simple as:

    rcr::detection_model rcr_model = rcr::load_detection_model("face_landmarks_model_rcr_22.bin");
    cv::Rect facebox = ...; // has to be detected by e.g. V&J
    LandmarkCollection<cv::Vec2f> landmarks = rcr_model.detect(image, facebox);

The full example is at [apps/rcr/rcr-detect.cpp](https://github.com/patrikhuber/superviseddescent/blob/master/apps/rcr/rcr-detect.cpp) and it is built with the library.

### Simple example for generic optimisation - approximate sin(x):

The following examples describe using simple hello-world example how to use the library for generic function optimisation (for example for other computer vision tasks). We'll start with a simple function, `sin(x)`.
Define a function:

    auto h = [](Mat value, size_t, int) { return std::sin(value.at<float>(0)); };

Generate training data (see `examples/simple_function.cpp` for the (in this case boring) code):
* training labels `y_tr` in the interval [-1:0.2:1]
* the inverse function values (the ground truth parameters) `x_tr`
* fixed starting values of the parameters `x0` (a constant value in this case).

Construct and train the model, and (optionally) specify a callback function that prints the residual after each learned regressor:

~~~{.cpp}
	vector<LinearRegressor> regressors(10);
	SupervisedDescentOptimiser<LinearRegressor> supervised_descent_model(regressors);
	auto print_residual = [&x_tr](const Mat& current_predictions) {
		cout << cv::norm(current_predictions, x_tr, cv::NORM_L2) / cv::norm(x_tr, cv::NORM_L2) << endl;
	};
	supervised_descent_model.train(x_tr, x0, y_tr, h, print_residual);
~~~	

The model can be tested on test data like so:
~~~{.cpp}
	Mat predictions = supervised_descent_model.test(x0_ts, y_ts, h);
	cout << "Test residual: " << cv::norm(predictions, x_ts_gt, cv::NORM_L2) / cv::norm(x_ts_gt, cv::NORM_L2) << endl;
~~~

Predictions on new data can similarly be made with:
~~~{.cpp}
SupervisedDescentOptimiser::predict(Mat x0, Mat y, H h)
~~~
which returns the prediction result.


### Pose estimation:

Using the `SupervisedDescentOptimiser` for 3D pose estimation from 2D landmarks works exactly in the same way.

* `h` is the projection function that projects the 3D model to 2D coordinates, given the current parameters
* `y` are the 2D landmarks, known (or detected) at training and testing time
* `x` are the rotation and translation parameters (the 6 DOF of the model)
* the initial parameters `x0` are all set to 0, with the exception of t_z being -2000.

~~~{.cpp}
class ModelProjection
{
public:
	ModelProjection(Mat model) : model(model) {};

	Mat operator()(Mat parameters, size_t regressor_level, int training_index = 0)
	{
		// parameters is a vector consisting of [R_x, R_y, R_z, t_x, t_y, t_z]
		projected_points = project_model(parameters);
		return projected_points;
	};
private:
	Mat model;
}
~~~

~~~{.cpp}
ModelProjection projection(facemodel);
supervised_descent_model.train(x_tr, x0, y_tr, projection, print_residual);
~~~

Testing and prediction work analogous.

For the documented full working example, see `examples/landmark_detection.cpp`.

### Landmark detection hello-world:

In contrast to the RCR, this is a simple-as-possible hello-world example to understand how the library works and to build your own applications.

The `SupervisedDescentOptimiser` can be used in the same way as before for landmark detection. In this case,

* `h` is a feature transform that extracts image features like HOG or SIFT from the image (we thus make it a function object, to store the images)
* we don't use the `y` values (the so called _template_) to train, because at testing, the HoG descriptors differ for each person (i.e. each persons face looks different)
* the parameters `x` are the current 2D landmark locations
* the initial parameters `x0` are computed by aligning the mean landmark to a detected face box.

~~~{.cpp}
class HogTransform
{
public:
	HogTransform(vector<Mat> images, ...HoG parameters...) { ... };
	
	Mat operator()(Mat parameters, size_t regressor_level, int training_index = 0)
	{
		// shortened, to get the idea across:
		Mat hog_descriptors = extract_HOG_features(images[training_index], parameters);
		return hog_descriptors;
	}
private:
	vector<Mat> images;
}
~~~

Training the model is the same, except we pass an empty `cv::Mat` instead of `y` values:
~~~{.cpp}
HogTransform hog(training_images, ...HoG parameters...);
supervised_descent_model.train(training_landmarks, x0, Mat(), hog, print_residual);
~~~

Testing and prediction work analogous.

For the documented full working example, see `examples/landmark_detection.cpp`.

## Build the examples and tests

Building of the examples and tests requires CMake>=2.8.11, OpenCV (core, imgproc, highgui, objdetect) and Boost (system, filesystem, program_options).

* copy `initial_cache.cmake.template` to `initial_cache.cmake`, edit the necessary paths
* create a build directory next to the `superviseddescent` folder: `mkdir build; cd build`
* `cmake -C ../superviseddescent/initial_cache.cmake -G "<your favourite generator>" ../superviseddescent -DCMAKE_INSTALL_PREFIX=install/`
* build using your favourite tools, e.g. `make; make install` or open the solution in Visual Studio.


## Documentation

Doxygen: http://patrikhuber.github.io/superviseddescent/doc/

The [examples](https://github.com/patrikhuber/superviseddescent/tree/master/examples) and the [Class List](http://patrikhuber.github.io/superviseddescent/doc/annotated.html) in doxygen are a good place to start.

## License & contributions

This code is licensed under the Apache License, Version 2.0

Contributions are very welcome! (best in the form of pull requests.) Please use Github issues for any bug reports, ideas, and discussions.

If you use this code in your own work, please cite the following paper: _Fitting 3D Morphable Models using Local Features_, P. Huber, Z. Feng, W. Christmas, J. Kittler, M. Rätsch, IEEE International Conference on Image Processing (ICIP) 2015, Québec City, Canada (http://arxiv.org/abs/1503.02330).
