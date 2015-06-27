project(rcr)
cmake_minimum_required(VERSION 2.8.12)

# The rcr app needs a few additional dependencies (e.g. boost filesystem and OpenCV highgui):
find_package(OpenCV 2.4.3 REQUIRED core imgproc highgui objdetect)
message(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")
# This allows us to compile in RelWithDebInfo. It'll use the Release-version of OpenCV:
set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)

find_package(Boost 1.48.0 COMPONENTS system filesystem program_options REQUIRED)
if(Boost_FOUND)
  message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
else(Boost_FOUND)
  message(FATAL_ERROR "Boost not found")
endif()

#include_directories(${superviseddescent_INCLUDE_DIR})
#include_directories(${CEREAL_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})

# Robust regression copse landmark detection

# Standard training:
add_executable(rcr-train rcr-train.cpp)
target_link_libraries(rcr-train ${OpenCV_LIBS} ${Boost_LIBRARIES})
set_target_properties(rcr-train PROPERTIES FOLDER "apps/rcr")
target_compile_options(rcr-train PUBLIC "$<$<CXX_COMPILER_ID:MSVC>:/openmp>") # adds /openmp on MSVC
target_compile_options(rcr-train PUBLIC "$<$<CXX_COMPILER_ID:GNU>:-fopenmp>") # same for gcc
target_compile_options(rcr-train PUBLIC "$<$<CXX_COMPILER_ID:Clang>:-fopenmp>") # and clang. Clang additionally needs an OpenMP runtime when running the executable.
target_link_libraries(rcr-train "$<$<CXX_COMPILER_ID:GNU>:-fopenmp>") # gcc and clang need a linker flag as well
target_link_libraries(rcr-train "$<$<CXX_COMPILER_ID:Clang>:-fopenmp>")

# Detect landmarks on an image:
add_executable(rcr-detect rcr-detect.cpp)
target_link_libraries(rcr-detect ${OpenCV_LIBS} ${Boost_LIBRARIES})
set_target_properties(rcr-detect PROPERTIES FOLDER "apps/rcr")

# Run landmark tracking on a video:
add_executable(rcr-track rcr-track.cpp)
target_link_libraries(rcr-track ${OpenCV_LIBS} ${Boost_LIBRARIES})
set_target_properties(rcr-track PROPERTIES FOLDER "apps/rcr")

# Install targets:
install(TARGETS rcr-train DESTINATION bin)
install(TARGETS rcr-detect DESTINATION bin)
install(TARGETS rcr-track DESTINATION bin)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/apps/rcr/data/ DESTINATION bin/data/rcr)
