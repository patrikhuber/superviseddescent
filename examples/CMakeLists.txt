project(examples)
cmake_minimum_required(VERSION 2.8.7)

# The examples need a few additional dependencies (e.g. boost filesystem and OpenCV highgui):
find_package(OpenCV 2.4.3 REQUIRED core imgproc highgui objdetect)
message(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

find_package(Boost 1.48.0 COMPONENTS system filesystem program_options REQUIRED)
if(Boost_FOUND)
  message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
else(Boost_FOUND)
  message(FATAL_ERROR "Boost not found")
endif()

include_directories(${Boost_INCLUDE_DIRS})

# Simple example:
add_executable(simple_function simple_function.cpp)
target_link_libraries(simple_function ${OpenCV_LIBS})
set_target_properties(simple_function PROPERTIES FOLDER "examples")

# Landmark detection example:
set(SOURCE
	landmark_detection.cpp
)
set(HEADERS
	hog.h
	#hog.c # used as a header, we don't want CMake to see it as a file it has to compile
)	
add_executable(landmark_detection ${SOURCE} ${HEADERS})
target_link_libraries(landmark_detection ${OpenCV_LIBS} ${Boost_LIBRARIES})
set_target_properties(landmark_detection PROPERTIES FOLDER "examples")

# Pose estimation example:
add_executable(pose_estimation pose_estimation.cpp)
target_link_libraries(pose_estimation ${OpenCV_LIBS} ${Boost_LIBRARIES})
set_target_properties(pose_estimation PROPERTIES FOLDER "examples")

# install target:
install(TARGETS simple_function DESTINATION bin)

install(TARGETS landmark_detection DESTINATION bin)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/examples/data/ DESTINATION bin/data/examples)

install(TARGETS pose_estimation DESTINATION bin)
