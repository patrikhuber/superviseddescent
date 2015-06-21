# Unit tests
# ==========
# Note: All the variables in here are inherited from the parent CMakeLists
find_package(OpenCV 2.4.3 REQUIRED core)
message(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

find_package(Boost 1.48.0 REQUIRED) # we only need the headers here
if(Boost_FOUND)
  message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
else(Boost_FOUND)
  message(FATAL_ERROR "Boost not found")
endif()

include_directories(${gtest_SOURCE_DIR}/include)
include_directories(${Boost_INCLUDE_DIRS}) # for erf.hpp

set(TEST_LINREG1D test_LinearRegressor1D)
add_executable(${TEST_LINREG1D} test_LinearRegressor1D.cpp)
target_link_libraries(${TEST_LINREG1D} ${OpenCV_LIBS} ${Boost_LIBRARIES} gtest_main gtest)
add_test(${TEST_LINREG1D} ${TEST_LINREG1D}) # Syntax: testname exename
set_target_properties(${TEST_LINREG1D} PROPERTIES FOLDER "tests")

set(TEST_LINREGND test_LinearRegressorND)
add_executable(${TEST_LINREGND} test_LinearRegressorND.cpp)
target_link_libraries(${TEST_LINREGND} ${OpenCV_LIBS} ${Boost_LIBRARIES} gtest_main gtest)
add_test(${TEST_LINREGND} ${TEST_LINREGND})
set_target_properties(${TEST_LINREGND} PROPERTIES FOLDER "tests")

set(TEST_SDO test_SupervisedDescentOptimiser)
add_executable(${TEST_SDO} test_SupervisedDescentOptimiser.cpp)
target_link_libraries(${TEST_SDO} ${OpenCV_LIBS} ${Boost_LIBRARIES} gtest_main gtest)
add_test(${TEST_SDO} ${TEST_SDO})
set_target_properties(${TEST_SDO} PROPERTIES FOLDER "tests")
