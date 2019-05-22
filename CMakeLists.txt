cmake_minimum_required(VERSION 3.0.0)
project("Hungarian Algorithm")

find_package(Eigen)

set(CMAKE_CXX_STANDARD 11)

add_library(hungarian
  src/hungarian.cpp
  )
target_include_directories(hungarian PUBLIC 
  include
  ${EIGEN_INCLUDE_DIR}
  )
