cmake_minimum_required(VERSION 2.8)
PROJECT(MetaObject)
CMAKE_POLICY(SET CMP0020 NEW)
include("${CMAKE_CURRENT_LIST_DIR}/cmake/CMakeMacros.cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(link_libs "")

set(Boost_required_components thread log log_setup)
find_package(PythonLibs QUIET)
if(PythonLibs_FOUND)
  set(Boost_required_components "${Boost_required_components} python")
  list(APPEND link_libs "${PYTHON_LIBRAY}")
endif()

find_package(Boost REQUIRED COMPONENTS ${Boost_required_components})
  include_directories(${Boost_INCLUDE_DIRS})
  IF(NOT WIN32)
    ADD_DEFINITIONS(-DBOOST_ALL_NO_LIB)
  ENDIF(NOT WIN32)

file(GLOB_RECURSE src "src/*.cpp")
file(GLOB_RECURSE hdr "include/*.h" "include/*.hpp")
include_directories("${CMAKE_CURRENT_LIST_DIR}/include")
GroupSources(include)
GroupSources(src)
  
add_library(MetaObject SHARED ${src} ${hdr})
