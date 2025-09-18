cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

# ----------------------- Ensure Eigen submodule is present -----------------------
# If building from a Git checkout and Eigen headers are missing, run:
#   git submodule update --init --recursive
set(EIGEN_SUBMODULE_DIR "${CMAKE_SOURCE_DIR}/vendor/eigen")
set(EIGEN_CORE_HEADER   "${EIGEN_SUBMODULE_DIR}/Eigen/Core")

if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  if(NOT EXISTS "${EIGEN_CORE_HEADER}")
    message(STATUS "Eigen submodule not found; initializing submodules...")
    find_program(GIT_EXECUTABLE git)
    if(NOT GIT_EXECUTABLE)
      message(FATAL_ERROR "git not found but Eigen submodule is missing. "
                          "Install git or run: git submodule update --init --recursive")
    endif()
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" submodule update --init --recursive
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      RESULT_VARIABLE _submod_res
      OUTPUT_VARIABLE _submod_out
      ERROR_VARIABLE  _submod_err
    )
    if(NOT _submod_res EQUAL 0)
      message(FATAL_ERROR "Failed to initialize submodules:\n${_submod_out}\n${_submod_err}")
    endif()
  endif()
endif()

# Final check (covers sdist/wheel path too)
if(NOT EXISTS "${EIGEN_CORE_HEADER}")
  message(FATAL_ERROR
    "Eigen headers not found at ${EIGEN_SUBMODULE_DIR}. "
    "If building from a Git checkout, run:\n  git submodule update --init --recursive")
endif()

# ------------------------------ Dependencies ------------------------------------
set(PYBIND11_NEWPYTHON ON)    # pybind11 multi-version helpers
find_package(pybind11 CONFIG REQUIRED)

# OpenMP (optional)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
endif()

# ------------------------------ Targets -----------------------------------------
# Common include dirs for our extensions
set(GEORGE_SRC_INCLUDE "${CMAKE_SOURCE_DIR}/src/george/include")

# kernel_interface module
pybind11_add_module(kernel_interface "src/george/kernel_interface.cpp")
target_include_directories(kernel_interface
  PRIVATE
    "${GEORGE_SRC_INCLUDE}"
    "${EIGEN_SUBMODULE_DIR}"
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(kernel_interface PRIVATE OpenMP::OpenMP_CXX)
endif()

# _hodlr module
pybind11_add_module(_hodlr "src/george/solvers/_hodlr.cpp")
target_include_directories(_hodlr
  PRIVATE
    "${GEORGE_SRC_INCLUDE}"
    "${EIGEN_SUBMODULE_DIR}"
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(_hodlr PRIVATE OpenMP::OpenMP_CXX)
endif()

# ------------------------------ Install layout ----------------------------------
# scikit-build-core will place these under wheel.install-dir ("george").
# `DESTINATION .` -> <wheel_root>/george/
install(TARGETS kernel_interface LIBRARY DESTINATION .)
install(TARGETS _hodlr           LIBRARY DESTINATION ./solvers)