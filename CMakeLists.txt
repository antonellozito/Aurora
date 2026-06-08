cmake_minimum_required(VERSION 3.17.2...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C Fortran)

set(Python_FIND_VIRTUALENV FIRST)
set(Python_FIND_STRATEGY LOCATION)

# When reusing a build directory across conda/venv environments, CMake may keep
# a stale cached Python executable. Reset that cache if the active environment
# has changed, so direct `cmake ..` builds follow the shell's current Python.
set(_ACTIVE_PYTHON_PREFIX "")
if(DEFINED ENV{VIRTUAL_ENV} AND NOT "$ENV{VIRTUAL_ENV}" STREQUAL "")
  set(_ACTIVE_PYTHON_PREFIX "$ENV{VIRTUAL_ENV}")
elseif(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
  set(_ACTIVE_PYTHON_PREFIX "$ENV{CONDA_PREFIX}")
endif()

if(_ACTIVE_PYTHON_PREFIX)
  set(Python_ROOT_DIR "${_ACTIVE_PYTHON_PREFIX}")
  if(DEFINED Python_EXECUTABLE)
    string(FIND "${Python_EXECUTABLE}" "${_ACTIVE_PYTHON_PREFIX}/" _PYTHON_IN_ACTIVE_ENV)
    if(NOT _PYTHON_IN_ACTIVE_ENV EQUAL 0)
      message(STATUS "Resetting cached Python from ${Python_EXECUTABLE} to match active environment ${_ACTIVE_PYTHON_PREFIX}")
      unset(Python_EXECUTABLE CACHE)
      unset(Python_ROOT_DIR CACHE)
      unset(Python_INCLUDE_DIR CACHE)
      unset(Python_INCLUDE_DIRS CACHE)
      unset(Python_LIBRARY CACHE)
      unset(Python_LIBRARIES CACHE)
      unset(Python_NumPy_INCLUDE_DIRS CACHE)
      unset(Python_NumPy_VERSION CACHE)
      set(Python_ROOT_DIR "${_ACTIVE_PYTHON_PREFIX}")
    endif()
  endif()
endif()

find_package(
  Python
  COMPONENTS Interpreter Development.Module NumPy
  REQUIRED)

message(STATUS "Using Python executable: ${Python_EXECUTABLE}")
message(STATUS "Using NumPy include dirs: ${Python_NumPy_INCLUDE_DIRS}")

# F2PY headers from same interpreter selected by find_package(Python).
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import numpy.f2py; print(numpy.f2py.get_include())"
  OUTPUT_VARIABLE F2PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

add_library(fortranobject OBJECT "${F2PY_INCLUDE_DIR}/fortranobject.c")
target_link_libraries(fortranobject PUBLIC Python::NumPy)
target_include_directories(fortranobject PUBLIC "${F2PY_INCLUDE_DIR}")
set_property(TARGET fortranobject PROPERTY POSITION_INDEPENDENT_CODE ON)


add_custom_command(
  OUTPUT _auroramodule.c example-f2pywrappers.f
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/aurora/main.f90"
          "${CMAKE_CURRENT_SOURCE_DIR}/aurora/impden.f90"
          "${CMAKE_CURRENT_SOURCE_DIR}/aurora/math.f90"
          "${CMAKE_CURRENT_SOURCE_DIR}/aurora/grids.f90"
  VERBATIM
  COMMAND "${Python_EXECUTABLE}" -m numpy.f2py
          "${CMAKE_CURRENT_SOURCE_DIR}/aurora/main.f90"
          "${CMAKE_CURRENT_SOURCE_DIR}/aurora/impden.f90"
          "${CMAKE_CURRENT_SOURCE_DIR}/aurora/math.f90"
          "${CMAKE_CURRENT_SOURCE_DIR}/aurora/grids.f90"
          -m _aurora --lower)

python_add_library(_aurora MODULE "${CMAKE_CURRENT_BINARY_DIR}/_auroramodule.c"
                   "${CMAKE_CURRENT_SOURCE_DIR}/aurora/main.f90"
                   "${CMAKE_CURRENT_SOURCE_DIR}/aurora/impden.f90"
                   "${CMAKE_CURRENT_SOURCE_DIR}/aurora/math.f90"
                   "${CMAKE_CURRENT_SOURCE_DIR}/aurora/grids.f90"
                   WITH_SOABI)
target_link_libraries(_aurora PRIVATE fortranobject)

install(TARGETS _aurora DESTINATION aurora/)
