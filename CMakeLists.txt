project(superviseddescent)
cmake_minimum_required(VERSION 2.8.7)
set(superviseddescent_VERSION_MAJOR 0)
set(superviseddescent_VERSION_MINOR 4)
set(superviseddescent_VERSION_PATCH 1)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# g++ needs a compiler flag to enable C++11 support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-std=c++11 HAS_CXX11_FLAG)
if (HAS_CXX11_FLAG)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

# All the options for building the library. Can be changed on the command-line or in initial_cache.cmake.
message(STATUS "Options:")
option(BUILD_TESTS "Builds the unit tests." OFF)
message(STATUS "BUILD_TESTS: ${BUILD_TESTS}")
option(BUILD_EXAMPLES "Build the example hello-world applications." ON)
message(STATUS "BUILD_EXAMPLES: ${BUILD_EXAMPLES}")
option(BUILD_APPS "Build the RCR landmark detection applications." ON)
message(STATUS "BUILD_APPS: ${BUILD_APPS}")
option(BUILD_DOCUMENTATION "Build the library documentation." OFF)
message(STATUS "BUILD_DOCUMENTATION: ${BUILD_DOCUMENTATION}")

# Build a CPack driven installer package:
include(InstallRequiredSystemLibraries) # This module will include any runtime libraries that are needed by the project for the current platform
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_VERSION_MAJOR "${superviseddescent_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${superviseddescent_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${superviseddescent_VERSION_PATCH}")
include(CPack)

# Find dependencies:
find_package(OpenCV 2.4.3 REQUIRED core)
message(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 found: ${EIGEN3_FOUND}")
message(STATUS "Eigen3 include dir found at ${EIGEN3_INCLUDE_DIR}")
message(STATUS "Eigen3 version: ${EIGEN3_VERSION}")

set(CEREAL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/3rdparty/cereal-1.1.1/include")

# Header files:
set(HEADERS
	include/superviseddescent/superviseddescent.hpp
	include/superviseddescent/regressors.hpp
	include/superviseddescent/verbose_solver.hpp
	include/superviseddescent/utils/mat_cerealisation.hpp
	include/superviseddescent/utils/mat_serialization.hpp
	include/superviseddescent/utils/ThreadPool.h
	include/rcr/adaptive_vlhog.hpp
	include/rcr/landmark.hpp
	include/rcr/landmarks_io.hpp
	include/rcr/helpers.hpp
	include/rcr/model.hpp
	include/rcr/hog.h # we don't add hog.c here because we don't want CMake to see it as a file it has to compile
)

# Add header includes:
include_directories("include")
include_directories(${CEREAL_INCLUDE_DIR})
#include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

# Custom target for the library, to make the headers show up in IDEs:
add_custom_target(superviseddescent SOURCES ${HEADERS})
source_group(superviseddescent include/superviseddescent/*)
source_group(superviseddescent\\utils include/superviseddescent/utils/*)
source_group(rcr include/rcr/*)

# The install target:
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include) # our library headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/3rdparty/cereal-1.1.1/include/ DESTINATION 3rdparty/cereal-1.1.1/include) # cereal headers

if(BUILD_TESTS)
	message(STATUS "Configuring gtest-1.7.0...")
	set(gtest_force_shared_crt ON CACHE BOOL "Use shared (DLL) run-time lib (CRT) even when Google Test is built as static lib.") # THIS IS NEWEST
	add_subdirectory(3rdparty/gtest-1.7.0)
	message(STATUS "Finished configuring gtest.")
	enable_testing() # Enables CTest
	add_custom_target(check-dbg COMMAND ${CMAKE_CTEST_COMMAND} -V -C Debug) # to get detailed output on VS
	add_custom_target(check-rel COMMAND ${CMAKE_CTEST_COMMAND} -V -C Release)
	add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

if(BUILD_APPS)
	add_subdirectory(apps/rcr)
endif()

if(BUILD_DOCUMENTATION)
	add_subdirectory(doc)
endif()
