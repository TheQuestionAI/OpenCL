cmake_minimum_required(VERSION 3.24)
project(another_test)

set(CMAKE_CXX_STANDARD 14)

add_executable(another_test main.cpp)

find_package(OpenCL REQUIRED)

target_link_libraries(another_test PRIVATE OpenCL::OpenCL)