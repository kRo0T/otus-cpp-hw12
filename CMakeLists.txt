cmake_minimum_required(VERSION 3.0.0)
project(hw12 VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(hw12 main.cpp)

include_directories(./eigen-3.4.0/eigen-3.4.0/)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
