cmake_minimum_required(VERSION 2.8)
PROJECT(MetaObject)
CMAKE_POLICY(SET CMP0020 NEW)
include("${CMAKE_CURRENT_LIST_DIR}/cmake/CMakeMacros.cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(link_libs "")
set(PROJECT_BIN_DIRS_DEBUG "")
set(PROJECT_BIN_DIRS_RELWITHDEBINFO "")
set(PROJECT_BIN_DIRS_RELEASE "")

set(Boost_required_components thread log log_setup unit_test_framework)
find_package(PythonLibs QUIET)
if(PythonLibs_FOUND)
  set(Boost_required_components "${Boost_required_components};python")
  list(APPEND link_libs "${PYTHON_LIBRAY}")
endif()
SET(CMAKE_CXX_FLAGS "/EHsc /FC /Zi /MP /Zm512 /bigobj /FS" CACHE STRING "")
SET(CMAKE_CXX_FLAGS_RELEASE "/Zo /d2Zi+ /Oy-" CACHE STRING "")
set(CMAKE_DEBUG_POSTFIX "d")
link_directories("${CMAKE_CURRENT_BINARY_DIR}")
file(GLOB_RECURSE src "src/*.cpp")
file(GLOB_RECURSE hdr "include/*.h" "include/*.hpp")
include_directories("${CMAKE_CURRENT_LIST_DIR}/include")

# ---------------------- External Dependencies ---------------------------------

# ---------------------- Boost ---------------------------------
find_package(Boost REQUIRED COMPONENTS ${Boost_required_components})
  include_directories(${Boost_INCLUDE_DIRS})
  link_directories(${Boost_LIBRARY_DIR_DEBUG})
  link_directories(${Boost_LIBRARY_DIR})
  list(APPEND PROJECT_BIN_DIRS_DEBUG ${Boost_LIBRARY_DIR})
  list(APPEND PROJECT_BIN_DIRS_RELEASE ${Boost_LIBRARY_DIR})
  list(APPEND PROJECT_BIN_DIRS_RELWITHDEBINFO ${Boost_LIBRARY_DIR})
  list(APPEND PROJECT_BIN_DIRS_DEBUG ${Boost_LIBRARY_DIR_DEBUG})
  list(APPEND PROJECT_BIN_DIRS_RELEASE ${Boost_LIBRARY_DIR_RELEASE})
  list(APPEND PROJECT_BIN_DIRS_RELWITHDEBINFO ${Boost_LIBRARY_DIR_RELEASE})
  IF(NOT WIN32)
    ADD_DEFINITIONS(-DBOOST_ALL_NO_LIB)
  ENDIF(NOT WIN32)

# ---------------------- cereal ---------------------------------  
find_package(cereal REQUIRED)
if(cereal_FOUND)
  include_directories(${cereal_INCLUDE_DIRS})
endif()


# ---------------------- Optional dependencies ------------------
find_package(Qt5 QUIET COMPONENTS Core Widgets Gui)
if(Qt5_FOUND)
  message("Building with qt support")
  add_definitions(-DHAVE_QT5)
  QT5_WRAP_CPP(MOC "include/MetaObject/Parameters/UI/Qt/SignalProxy.hpp")
  list(APPEND src ${MOC})
  INCLUDE_DIRECTORIES(${Qt5Gui_INCLUDE_DIRS} ${Qt5Widgets_INCLUDE_DIRS} ${Qt5Core_INCLUDE_DIRS})
  list(APPEND link_libs Qt5::Core Qt5::Widgets Qt5::Gui)
else()
  message("Building without qt support")
endif()

find_package(OpenCV QUIET COMPONENTS core)
if(OpenCV_FOUND)
  message("Building with opencv support")
  add_definitions(-DHAVE_OPENCV)
  include_directories(${OpenCV_INCLUDE_DIRS})
else()
  message("Building without opencv support")
endif()


# ---------------------- rcc ---------------------------------  
SET(IObject_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(RCC_INCLUDE_DEPENDENCIES "${Boost_INCLUDE_DIRS};${CMAKE_CURRENT_LIST_DIR}/include")
set(RCC_LIBRARY_DIRS_DEBUG "${Boost_LIBRARY_DIR};${Boost_LIBRARY_DIR_DEBUG};${CMAKE_CURRENT_BINARY_DIR}/Debug")
set(RCC_LIBRARY_DIRS_RELWITHDEBINFO "${Boost_LIBRARY_DIR};${Boost_LIBRARY_DIR_DEBUG};${CMAKE_CURRENT_BINARY_DIR}/RelWithDebInfo")
ADD_SUBDIRECTORY("dependencies/rcc")
include_directories(${RCC_INCLUDE_DIRS})
set_target_properties(RuntimeCompiler PROPERTIES FOLDER Dependencies)
set_target_properties(RuntimeObjectSystem PROPERTIES FOLDER Dependencies)

GroupSources(include)
GroupSources(src)
  
add_library(MetaObject SHARED ${src} ${hdr})
target_link_libraries(MetaObject RuntimeObjectSystem RuntimeCompiler ${link_libs})

# -------------------- instantiations --------------------------
file(GLOB_RECURSE instantiations_src "instantiations/*.cpp")
include_directories("instantiations")
add_library(instantiations SHARED ${instantiations_src})
target_link_libraries(instantiations MetaObject)
if(Qt5_FOUND)
  target_link_libraries(instantiations Qt5::Core Qt5::Gui Qt5::Widgets)
endif()
if(OpenCV_FOUND)
  target_link_libraries(instantiations ${OpenCV_LIBS})
endif()


# ----------------------- tests -----------------------------------
if(Boost_UNIT_TEST_FRAMEWORK_FOUND)
  file(GLOB_RECURSE test_srcs "tests/*.cpp")
  include(CTest)
  enable_testing()
  foreach(fn ${test_srcs})
    get_filename_component(name ${fn} NAME_WE)
    add_executable(${name} "${fn}")
    TARGET_LINK_LIBRARIES(${name} MetaObject ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
    ADD_DEPENDENCIES(${name} MetaObject)
    set_target_properties(${name} PROPERTIES FOLDER Tests/MetaObject)
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests)
    add_test(${name} ${name})
    if(MSVC)
      CONFIGURE_FILE("tests/Test.vcxproj.user.in" ${CMAKE_BINARY_DIR}/${name}.vcxproj.user @ONLY)
    endif()
  endforeach()
endif()

# ----------------------- examples ---------------------------
ADD_SUBDIRECTORY("examples")