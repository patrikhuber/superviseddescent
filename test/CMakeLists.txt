# Unit tests
# ==========
# Note: All the variables in here are inherited from the parent CMakeLists
find_package(OpenCV 2.4.3 REQUIRED core)
message(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

include_directories(${gtest_SOURCE_DIR}/include)

set(TEST_LINREG1D test_LinearRegressor1D)
add_executable(${TEST_LINREG1D} test_LinearRegressor1D.cpp)
target_link_libraries(${TEST_LINREG1D} ${OpenCV_LIBS} ${Boost_LIBRARIES} gtest_main gtest)
add_test(${TEST_LINREG1D} ${TEST_LINREG1D}) # Syntax: testname exename

set(TEST_LINREGND test_LinearRegressorND)
add_executable(${TEST_LINREGND} test_LinearRegressorND.cpp)
target_link_libraries(${TEST_LINREGND} ${OpenCV_LIBS} ${Boost_LIBRARIES} gtest_main gtest)
add_test(${TEST_LINREGND} ${TEST_LINREGND})

set(TEST_SDO test_SupervisedDescentOptimiser)
add_executable(${TEST_SDO} test_SupervisedDescentOptimiser.cpp)
target_link_libraries(${TEST_SDO} ${OpenCV_LIBS} ${Boost_LIBRARIES} gtest_main gtest)
add_test(${TEST_SDO} ${TEST_SDO})
