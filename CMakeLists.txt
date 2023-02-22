cmake_minimum_required(VERSION 3.0.0)
set(PATCH_VERSION "1" CACHE INTERNAL "Patch version")
set(PROJECT_VESRION 0.0.${PATCH_VERSION})
set(ENABLE_PRECOMPILED_HEADERS OFF)

project(fashion_mnist VERSION ${PROJECT_VESRION})

#include(CTest)
#enable_testing()

add_executable(fashion_mnist main.cpp)

add_subdirectory(eigen-3.4.0/eigen-3.4.0/)
#include_directories(./eigen-3.4.0/eigen-3.4.0/)
#set(Eigen3_DIR "./eigen-3.4.0/eigen-3.4.0/cmake/")
#set(CMAKE_PREFIX_PATH "./eigen-3.4.0/eigen-3.4.0/")
#find_package(Eigen3 3.4.0 REQUIRED NO_MODULE)

target_include_directories(fashion_mnist PRIVATE ${PROJECT_SOURCE_DIR}/eigen-3.4.0/eigen-3.4.0/)

# Link against the Eigen library
#target_link_libraries(fashion_mnist Eigen)

set(CPACK_GENERATOR DEB)
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(CPACK_PACKAGE_CONTACT e.krotov@mail.ru)
include(CPack)
