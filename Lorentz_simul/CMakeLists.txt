cmake_minimum_required (VERSION 2.6)
project (Tutorial)
find_package(Armadillo REQUIRED)

#include_directories(${ARMADILLO_INCLUDE_DIRS})
#target_link_libraries(${ARMADILLO_LIBRARIES})

add_executable(Tutorial main.cxx)

