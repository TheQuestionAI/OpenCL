cmake_minimum_required(VERSION 3.24)
project(another_test)

set(CMAKE_CXX_STANDARD 17)

#include_directories(include)

add_executable(OpenCL_Kernel_Test_Framework main.cpp)

find_package(OpenCL REQUIRED)

target_link_libraries(OpenCL_Kernel_Test_Framework PRIVATE OpenCL::OpenCL)