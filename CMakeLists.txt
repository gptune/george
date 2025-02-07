cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

include_directories(
  "src/george/include"
  "vendor/eigen"
)


find_package(OpenMP)
if(OPENMP_CXX_FOUND)
  set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
  set(OpenMP_CXX_FLAGS_EXPORT "${OpenMP_CXX_FLAGS}")
  set(CMAKE_CXX_IMPLICIT_LINK_LIBRARIES "${CMAKE_CXX_IMPLICIT_LINK_LIBRARIES} ${OpenMP_CXX_FLAGS}")
endif()



pybind11_add_module(kernel_interface "src/george/kernel_interface.cpp")
install(TARGETS kernel_interface LIBRARY DESTINATION .)

pybind11_add_module(_hodlr "src/george/solvers/_hodlr.cpp")
install(TARGETS _hodlr LIBRARY DESTINATION ./solvers)
