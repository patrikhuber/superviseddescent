# Todo: Set a VS folder. See CMake talk slides C++Now, hpx-guy

# The examples need a few additional dependencies (e.g. boost filesystem and OpenCV highgui):
find_package(OpenCV 2.4.3 REQUIRED core imgproc highgui objdetect)
message(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")
set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)

find_package(Boost 1.48.0 COMPONENTS system filesystem program_options serialization REQUIRED)
if(Boost_FOUND)
  message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
else(Boost_FOUND)
  message(FATAL_ERROR "Boost not found")
endif()

include_directories(${superviseddescent_INCLUDE_DIR})

# Robust regression copse landmark detection training:
add_executable(rcr-train rcr-train.cpp ../hog.c ../hog.h eos_core_landmark.hpp eos_io_landmarks.hpp adaptive_vlhog.hpp helpers.hpp model.hpp)
target_link_libraries(rcr-train ${OpenCV_LIBS} ${Boost_LIBRARIES})

# Robust regression copse landmark detection:
add_executable(rcr-detect rcr-detect.cpp ../hog.c ../hog.h eos_core_landmark.hpp eos_io_landmarks.hpp adaptive_vlhog.hpp helpers.hpp model.hpp)
target_link_libraries(rcr-detect ${OpenCV_LIBS} ${Boost_LIBRARIES})

# install targets:
install(TARGETS rcr-train DESTINATION bin)
#install(DIRECTORY ${CMAKE_SOURCE_DIR}/examples/data DESTINATION bin)

install(TARGETS rcr-detect DESTINATION bin)
