cmake_minimum_required(VERSION 3.0.0)
set(PATCH_VERSION "1" CACHE INTERNAL "Patch version")
set(PROJECT_VESRION 0.0.${PATCH_VERSION})

project(fashion_mnist VERSION ${PROJECT_VESRION})

include(CTest)
enable_testing()

add_executable(fashion_mnist main.cpp)

include_directories(./eigen-3.4.0/eigen-3.4.0/)

set(CPACK_GENERATOR DEB)
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(CPACK_PACKAGE_CONTACT e.krotov@mail.ru)
include(CPack)
