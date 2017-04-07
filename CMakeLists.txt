cmake_minimum_required(VERSION 2.8)
PROJECT(MetaObject)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
CMAKE_POLICY(SET CMP0020 NEW)
include("${CMAKE_CURRENT_LIST_DIR}/cmake/CMakeMacros.cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(link_libs "")

option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(WITH_QT "Enable QT support" ON)
option(WITH_PYTHON "Enable Python bindings" ON)
option(WITH_OPENCV "Enable OpenCV types" ON )
option(WITH_CUDA "Enable CUDA recompilation" ON)
option(WITH_WT "Enable WT support" ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include(cmake/OpenCVPCHSupport.cmake)
set(include_dirs_ "")
set(link_dirs_deb "")
set(link_dirs_rel "")
set(compile_flags "")

set(HAVE_PYTHON OFF)
if(WITH_PYTHON)
  find_package(PythonLibs QUIET)
  find_package(PythonInterp QUIET)
  if(PythonLibs_FOUND AND PythonInterp_FOUND)
    find_package(Boost COMPONENTS python)
    if(Boost_PYTHON_FOUND)
      set(HAVE_PYTHON ON)
      message("Building python bindings")
      list(APPEND link_libs "${Boost_PYTHON_LIBRARY_RELEASE};${PYTHON_LIBRARY}")
      get_filename_component(_python_lib_dir "${PYTHON_LIBRARY}" DIRECTORY)
      link_directories(${_python_lib_dir})
      include_directories(${PYTHON_INCLUDE_DIR})
      add_definitions(-DHAVE_BOOST_PYTHON)
    endif()
  endif()
endif()

if(WIN32)
  list(APPEND CMAKE_CXX_FLAGS "/EHsc /FC /Zi /MP /Zm512 /bigobj /FS")
  list(APPEND CMAKE_CXX_FLAGS_RELEASE "/Zo /d2Zi+ /Oy-")
  ADD_DEFINITIONS(-D_ENABLE_ATOMIC_ALIGNMENT_FIX)
else(WIN32)
  #list(APPEND CMAKE_CXX_FLAGS "--std=c++11")
  list(APPEND link_libs pthread)
  add_definitions(-D_MWAITXINTRIN_H_INCLUDED)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG")
endif(WIN32)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64.*|AARCH64.*)")
  set(AARCH64 1)
  list(APPEND link_libs atomic)
else()
  set(AARCH64 0)
endif()

set(CMAKE_DEBUG_POSTFIX "d")
link_directories("${CMAKE_CURRENT_BINARY_DIR}")
file(GLOB_RECURSE src "src/*.cpp")
file(GLOB_RECURSE knl "src/*.cu")
file(GLOB_RECURSE hdr "include/*.h" "include/*.hpp")
include_directories("${CMAKE_CURRENT_LIST_DIR}/include")

# ---------------------- External Dependencies ---------------------------------

# ---------------------- Boost ---------------------------------
set(Boost_required_components system thread log log_setup unit_test_framework)
set(Boost_USE_STATIC_LIBS        OFF)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME     OFF)
ADD_DEFINITIONS(-DBOOST_ALL_DYN_LINK)
find_package(Boost REQUIRED COMPONENTS ${Boost_required_components})
  include_directories(${Boost_INCLUDE_DIRS})
  link_directories(${Boost_LIBRARY_DIR_DEBUG})
  link_directories(${Boost_LIBRARY_DIR})
  list(APPEND link_libs ${Boost_LIBRARIES})
  list(APPEND PROJECT_BIN_DIRS_DEBUG_ ${Boost_LIBRARY_DIR})
  list(APPEND PROJECT_BIN_DIRS_RELEASE_ ${Boost_LIBRARY_DIR})
  list(APPEND PROJECT_BIN_DIRS_RELWITHDEBINFO_ ${Boost_LIBRARY_DIR})
  list(APPEND PROJECT_BIN_DIRS_DEBUG_ ${Boost_LIBRARY_DIR_DEBUG})
  list(APPEND PROJECT_BIN_DIRS_RELEASE_ ${Boost_LIBRARY_DIR_RELEASE})
  list(APPEND PROJECT_BIN_DIRS_RELWITHDEBINFO_ ${Boost_LIBRARY_DIR_RELEASE})
  set(bin_dirs_ "${BIN_DIRS};Boost")
  list(REMOVE_DUPLICATES bin_dirs_)
  set(BIN_DIRS "${bin_dirs_}" CACHE STRING "" FORCE)


# ---------------------- cereal ---------------------------------
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/dependencies/cereal/include")
set(MetaObject_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/cereal/include;${CMAKE_CURRENT_SOURCE_DIR}/MetaParameters" CACHE PATH "" FORCE)
set(PROJECT_BIN_DIRS_RELWITHDEBINFO "")
set(PROJECT_BIN_DIRS_RELEASE "")
set(PROJECT_BIN_DIRS_DEBUG "")

# CUDA
if(WITH_CUDA)
  FIND_PACKAGE(CUDA REQUIRED)
  include("cmake/Cuda.cmake")
  IF(${CUDA_FOUND})
    INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
    ADD_DEFINITIONS(-DHAVE_CUDA)
    list(APPEND PROJECT_BIN_DIRS_DEBUG_ ${CUDA_BIN_DIR})
    list(APPEND PROJECT_BIN_DIRS_RELEASE_ ${CUDA_BIN_DIR})
    list(APPEND PROJECT_BIN_DIRS_RELWITHDEBINFO_ ${CUDA_BIN_DIR})
    if(NOT WIN32)
      #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--std=c++11")
    endif(NOT WIN32)
  ENDIF(${CUDA_FOUND})
endif(WITH_CUDA)

# ---------------------- rcc ---------------------------------  
SET(IObject_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(RCC_INCLUDE_DEPENDENCIES "${Boost_INCLUDE_DIRS};${CMAKE_CURRENT_LIST_DIR}/include;${cereal_INCLUDE_DIRS}")
set(RCC_LIBRARY_DIRS_DEBUG "${Boost_LIBRARY_DIR};${Boost_LIBRARY_DIR_DEBUG};${CMAKE_BINARY_DIR}/Debug")
set(RCC_LIBRARY_DIRS_RELWITHDEBINFO "${Boost_LIBRARY_DIR};${Boost_LIBRARY_DIR_DEBUG};${CMAKE_BINARY_DIR}/RelWithDebInfo")
if(MSVC)
  set(RCC_COMPILE_FLAGS "/DBOOST_ALL_DYN_LINK")
else()
  set(RCC_COMPILE_FLAGS "-DBOOST_ALL_DYN_LINK")
endif()
ADD_SUBDIRECTORY("dependencies/rcc")
include_directories(${RCC_INCLUDE_DIRS})
set_target_properties(RuntimeCompiler PROPERTIES FOLDER Dependencies)
set_target_properties(RuntimeObjectSystem PROPERTIES FOLDER Dependencies)
SET(RCC_INCLUDE_DIRS "${RCC_INCLUDE_DIRS}" PARENT_SCOPE)
set(RCC_PERMODULE_INTERFACE_SOURCE ${RCC_PERMODULE_INTERFACE_SOURCE} PARENT_SCOPE)

set(MO_PROFILING_MODE "BUILTIN" CACHE STRING "Set MetaObject profiling mode")
set_property(CACHE MO_PROFILING_MODE PROPERTY STRINGS BUILTIN DYNAMIC NONE)
include_directories("dependencies/Remotery/lib")
find_package(NVToolsExt QUIET)
if(NVToolsExt)
  include_directories(${NV_TOOLS_EXT_INCLUDE_DIR})
  add_definitions(-DHAVE_NVTX)
endif()
if(${MO_PROFILING_MODE} STREQUAL "BUILTIN")
  add_definitions(-DRMT_BUILTIN)
  list(APPEND link_libs Remotery)
  include_directories("dependencies/Remotery/lib")
endif()
if(${MO_PROFILING_MODE} STREQUAL "NONE")
  add_definitions(-DPROFILING_NONE)
endif()

add_definitions(-DRMT_USE_CUDA)

# ---------------------- Optional dependencies ------------------
# Qt5
if(WITH_QT)
  find_package(Qt5 QUIET COMPONENTS Core Widgets Gui)
  if(Qt5_FOUND)
    set(AUTOMOC OFF)
    add_definitions(-DHAVE_QT5)
    QT5_WRAP_CPP(MOC "${CMAKE_CURRENT_LIST_DIR}/include/MetaObject/Parameters/UI/Qt/SignalProxy.hpp")
    list(APPEND src ${MOC})
    INCLUDE_DIRECTORIES(${Qt5Gui_INCLUDE_DIRS} ${Qt5Widgets_INCLUDE_DIRS} ${Qt5Core_INCLUDE_DIRS})
    list(APPEND link_libs Qt5::Core Qt5::Widgets Qt5::Gui)
    get_target_property(qt5_core_bin_ Qt5::Core IMPORTED_LOCATION_DEBUG)
    get_filename_component(qt5_bin_dir_ "${qt5_core_bin_}" DIRECTORY)
    list(APPEND PROJECT_BIN_DIRS_DEBUG ${qt5_bin_dir_})
    list(APPEND PROJECT_BIN_DIRS_RELEASE ${qt5_bin_dir_})
    list(APPEND PROJECT_BIN_DIRS_RELWITHDEBINFO ${qt5_bin_dir_})
    set(Qt5_BIN_DIR_OPT ${qt5_bin_dir_} CACHE PATH "" FORCE)
    set(Qt5_BIN_DIR ${qt5_bin_dir_} CACHE PATH "" FORCE)
    set(Qt5_BIN_DIR_DBG ${qt5_bin_dir_} CACHE PATH "" FORCE)
	set(bin_dirs_ "${BIN_DIRS};Qt5")
	list(REMOVE_DUPLICATES bin_dirs_)
	set(BIN_DIRS "${bin_dirs_}" CACHE STRING "" FORCE)
    rcc_find_path(Qt5_PLUGIN_PATH qwindows.dll PATHS ${Qt5_BIN_DIR}/../plugins/platforms)
  endif()
endif()

# WT
if(WITH_WT)
find_package(Wt QUIET)
  if(Wt_FOUND)
    add_definitions(-DHAVE_WT)
    include_directories(${Wt_INCLUDE_DIR})
    list(APPEND link_libs
      debug;${Wt_HTTP_DEBUG_LIBRARY}
      debug;${Wt_DEBUG_LIBRARY}
      optimized;${Wt_LIBRARY}
      optimized;${Wt_HTTP_LIBRARY}
    )

    find_path(Wt_BIN_DIR
      NAMES wt.dll wtd.dll
      PATHS ${Wt_DIR} "${Wt_INCLUDE_DIR}/../"
      ENV ${Wt_DIR}
      PATH PATH_SUFFIXES bin)

    if(Wt_BIN_DIR)
      set(Wt_BIN_DIR_DBG ${Wt_BIN_DIR} CACHE PATH "" FORCE)
      set(Wt_BIN_DIR_OPT ${Wt_BIN_DIR} CACHE PATH "" FORCE)
	  set(bin_dirs_ "${BIN_DIRS};Wt")
	  list(REMOVE_DUPLICATES bin_dirs_)
	  set(BIN_DIRS "${bin_dirs_}" CACHE STRING "" FORCE)
    endif(Wt_BIN_DIR)
  endif(Wt_FOUND)
endif(WITH_WT)

# OpenCV
if(WITH_OPENCV)
  find_package(OpenCV QUIET COMPONENTS core cudaarithm)
  if(OpenCV_FOUND)
    add_definitions(-DHAVE_OPENCV)
    include_directories(${OpenCV_INCLUDE_DIRS})
	
	if(WIN32)
		set(bin_dirs_ "${BIN_DIRS};OpenCV")
		list(REMOVE_DUPLICATES bin_dirs_)
		set(BIN_DIRS "${bin_dirs_}" CACHE STRING "" FORCE)
		list(REMOVE_DUPLICATES BIN_DIRS)
		get_target_property(dll_file_path_ opencv_core IMPORTED_LOCATION_RELEASE)
		get_filename_component(dll_path ${dll_file_path_} DIRECTORY)
		set(OpenCV_BIN_DIR ${dll_path} CACHE PATH "" FORCE)
		set(OpenCV_BIN_DIR_OPT ${dll_path} CACHE PATH "" FORCE)
		list(APPEND PROJECT_BIN_DIRS_DEBUG ${dll_path})
		list(APPEND PROJECT_BIN_DIRS_RELEASE ${dll_path})
		list(APPEND PROJECT_BIN_DIRS_RELWITHDEBINFO ${dll_path})
	endif()
	
  endif()
endif(WITH_OPENCV)

# ZeroMQ
find_package(ZeroMQ QUIET)
if(${ZeroMQ_FOUND})
  include_directories(${ZeroMQ_INCLUDE_DIR})
  include_directories("${CMAKE_CURRENT_LIST_DIR}/dependencies/cppzmq") # cpp bindings
  list(APPEND link_libs "optimized;${ZeroMQ_LIBRARY_RELEASE};debug;${ZeroMQ_LIBRARY_DEBUG}")
  add_definitions(-DHAVE_ZEROMQ)
endif(${ZeroMQ_FOUND})



add_subdirectory("dependencies/Remotery")
set_target_properties(Remotery PROPERTIES FOLDER Dependencies)
GroupSources(include)
GroupSources(src)
cuda_add_library(MetaObject SHARED ${src} ${hdr} ${knl} src/MetaObject/precompiled.hpp)

if(NOT CUDA_CUDA_LIBRARY)
    set(CUDA_CUDA_LIBRARY "/usr/local/nvidia/lib64/libcuda.so")
endif(NOT CUDA_CUDA_LIBRARY)

target_link_libraries(MetaObject
  RuntimeObjectSystem
  RuntimeCompiler
  ${link_libs}
  ${OpenCV_LIBS}
  ${CUDA_CUDART_LIBRARY}
  ${CUDA_CUDA_LIBRARY}
)
if(HAVE_PYTHON)
    target_link_libraries(MetaObject
        ${PYTHON_LIBRARY}
    )

endif(HAVE_PYTHON)

ocv_add_precompiled_header_to_target(MetaObject src/precompiled.hpp)

if(RCC_VERBOSE_CONFIG)
  message(STATUS "MetaObject compile flags\n"
                 "  CXX flags: ${CMAKE_CXX_FLAGS}\n"
                 "  CXX flags Release: ${CMAKE_CXX_FLAGS_RELEASE}\n"
                 "  CXX flags Debug: ${CMAKE_CXX_FLAGS_DEBUG}\n"
                 "  CUDA flags: ${CUDA_NVCC_FLAGS}\n"
                 "  CUDA flags Release: ${CUDA_NVCC_FLAGS_RELEASE}\n"
                 "  CUDA flags Debug: ${CUDA_NVCC_FLAGS_DEBUG}\n")
endif()


# -------------------- instantiations --------------------------
file(GLOB_RECURSE instantiations_src "MetaParameters/*")
include_directories("MetaParameters")
add_library(MetaParameters SHARED ${instantiations_src})
set_target_properties(MetaParameters PROPERTIES FOLDER Dependencies)
target_link_libraries(MetaParameters MetaObject)
if(Qt5_FOUND)
  target_link_libraries(MetaParameters Qt5::Core Qt5::Gui Qt5::Widgets)
endif()
if(OpenCV_FOUND AND WITH_OPENCV)
  target_link_libraries(MetaParameters ${OpenCV_LIBS})
endif()

MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()

MACRO(APPEND_BIN_DIR ARG1 ARG2)
  IF(EXISTS "${${ARG1}}" AND IS_DIRECTORY "${${ARG1}}")
    LIST(APPEND ${ARG2} ${${ARG1}})
  ELSE()
    if(RCC_VERBOSE_CONFIG)
      MESSAGE("${ARG1} : ${${ARG1}} is not a valid directory")
    endif()
  ENDIF()
ENDMACRO(APPEND_BIN_DIR)


FOREACH(dir ${BIN_DIRS})
  APPEND_BIN_DIR(${dir}_BIN_DIR_DBG PROJECT_BIN_DIRS_DEBUG)
  APPEND_BIN_DIR(${dir}_BIN_DIR_OPT PROJECT_BIN_DIRS_RELEASE)
  APPEND_BIN_DIR(${dir}_BIN_DIR_OPT PROJECT_BIN_DIRS_RELWITHDEBINFO)
ENDFOREACH(dir ${BIN_DIRS})

IF(Boost_LIBRARY_DIR_DEBUG AND EXISTS "${Boost_LIBRARY_DIR_DEBUG}" AND IS_DIRECTORY "${Boost_LIBRARY_DIR_DEBUG}")
 LIST(APPEND PROJECT_BIN_DIRS_DEBUG ${Boost_LIBRARY_DIR_DEBUG})
ELSE(Boost_LIBRARY_DIR_DEBUG AND EXISTS "${Boost_LIBRARY_DIR_DEBUG}" AND IS_DIRECTORY "${Boost_LIBRARY_DIR_DEBUG}")
  IF(BOOST_LIBRARYDIR AND EXISTS "${BOOST_LIBRARYDIR}" AND IS_DIRECTORY "${BOOST_LIBRARYDIR}")
    LIST(APPEND PROJECT_BIN_DIRS_DEBUG ${BOOST_LIBRARYDIR})
  ELSE(BOOST_LIBRARYDIR AND EXISTS "${BOOST_LIBRARYDIR}" AND IS_DIRECTORY "${BOOST_LIBRARYDIR}")
    MESSAGE("Could not find boost binary dir in Boost_LIBRARY_DIR_DEBUG or BOOST_LIBRARYDIR")
  ENDIF(BOOST_LIBRARYDIR AND EXISTS "${BOOST_LIBRARYDIR}" AND IS_DIRECTORY "${BOOST_LIBRARYDIR}")
ENDIF(Boost_LIBRARY_DIR_DEBUG AND EXISTS "${Boost_LIBRARY_DIR_DEBUG}" AND IS_DIRECTORY "${Boost_LIBRARY_DIR_DEBUG}")

IF(Boost_LIBRARY_DIR_RELEASE AND EXISTS "${Boost_LIBRARY_DIR_RELEASE}" AND IS_DIRECTORY "${Boost_LIBRARY_DIR_RELEASE}")
  LIST(APPEND PROJECT_BIN_DIRS_RELEASE ${Boost_LIBRARY_DIR_RELEASE})
  LIST(APPEND PROJECT_BIN_DIRS_RELWITHDEBINFO ${Boost_LIBRARY_DIR_RELEASE})
ELSE(Boost_LIBRARY_DIR_RELEASE AND EXISTS "${Boost_LIBRARY_DIR_RELEASE}" AND IS_DIRECTORY "${Boost_LIBRARY_DIR_RELEASE}")
  IF(BOOST_LIBRARYDIR AND EXISTS "${BOOST_LIBRARYDIR}" AND IS_DIRECTORY "${BOOST_LIBRARYDIR}")
    LIST(APPEND PROJECT_BIN_DIRS_RELEASE ${BOOST_LIBRARYDIR})
    LIST(APPEND PROJECT_BIN_DIRS_RELWITHDEBINFO ${BOOST_LIBRARYDIR})
  ELSE(BOOST_LIBRARYDIR AND EXISTS "${BOOST_LIBRARYDIR}" AND IS_DIRECTORY "${BOOST_LIBRARYDIR}")
    IF(Boost_LIBRARY_DIR)
      LIST(APPEND PROJECT_BIN_DIRS_RELEASE ${Boost_LIBRARY_DIR})
      LIST(APPEND PROJECT_BIN_DIRS_DEBUG ${Boost_LIBRARY_DIR})
    ELSE()
      MESSAGE("Could not find boost binary dir in Boost_LIBRARY_DIR_DEBUG or BOOST_LIBRARYDIR")
    ENDIF()
  ENDIF(BOOST_LIBRARYDIR AND EXISTS "${BOOST_LIBRARYDIR}" AND IS_DIRECTORY "${BOOST_LIBRARYDIR}")
ENDIF(Boost_LIBRARY_DIR_RELEASE AND EXISTS "${Boost_LIBRARY_DIR_RELEASE}" AND IS_DIRECTORY "${Boost_LIBRARY_DIR_RELEASE}")

list(REMOVE_DUPLICATES PROJECT_BIN_DIRS_DEBUG)
list(REMOVE_DUPLICATES PROJECT_BIN_DIRS_RELEASE)
list(REMOVE_DUPLICATES PROJECT_BIN_DIRS_RELWITHDEBINFO)

get_target_property(target_include_dirs_ MetaObject INCLUDE_DIRECTORIES)
get_target_property(target_link_libs_    MetaObject LINK_LIBRARIES)
set(LINK_DIRS_RELEASE "")
set(LINK_DIRS_DEBUG "")

foreach( lib ${target_link_libs_})
    if(EXISTS ${lib})
        GET_FILENAME_COMPONENT(dir ${lib} DIRECTORY)
        if(dir)
            if(RCC_VERBOSE_CONFIG)
                message(STATUS "Library ${lib} link directory: ${dir}")
            endif()
            list(APPEND LINK_DIRS_RELEASE ${dir})
            list(APPEND LINK_DIRS_DEBUG ${dir})
        endif()
    endif()
    if(TARGET ${lib})
        get_target_property(rel ${lib} IMPORTED_IMPLIB_RELEASE)
        get_target_property(_rel ${lib} IMPORTED_LOCATION_RELEASE)
        get_target_property(deb ${lib} IMPORTED_IMPLIB_DEBUG)
        get_target_property(_deb ${lib} IMPORTED_LOCATION_DEBUG)
        GET_FILENAME_COMPONENT(rel_ ${rel} DIRECTORY)
        GET_FILENAME_COMPONENT(deb_ ${deb} DIRECTORY)
        LIST(APPEND LINK_DIRS_DEBUG "${deb_}")
        LIST(APPEND LINK_DIRS_RELEASE "${rel_}")
    endif()
endforeach()
list(APPEND LINK_DIRS_RELEASE ${BOOST_LIBRARYDIR})
list(APPEND LINK_DIRS_DEBUG ${BOOST_LIBRARYDIR})
if(MSVC)
	list(APPEND LINK_DIRS_RELEASE "${CMAKE_BINARY_DIR}/Release")
	list(APPEND LINK_DIRS_DEBUG "${CMAKE_BINARY_DIR}/Debug")
endif()
LIST(REMOVE_DUPLICATES target_include_dirs_)
LIST(REMOVE_DUPLICATES LINK_DIRS_RELEASE)
LIST(REMOVE_DUPLICATES LINK_DIRS_DEBUG)

WRITE_RCC_CONFIG("${target_include_dirs_}" "${LINK_DIRS_DEBUG}" "${LINK_DIRS_RELEASE}" "")

# ----------------------- tests -----------------------------------
if(Boost_UNIT_TEST_FRAMEWORK_FOUND AND BUILD_TESTS)
  include(CTest)
  enable_testing()
  SUBDIRLIST(tests "${CMAKE_CURRENT_LIST_DIR}/tests")
  foreach(test ${tests})
    file(GLOB_RECURSE test_srcs "tests/${test}/*.cpp")
    IF(CUDA_FOUND)
      file(GLOB_RECURSE test_knl "tests/${test}/*.cu")
    ENDIF()
    LIST(LENGTH test_knl num_knl)
    IF(${num_knl} GREATER 0)
      cuda_add_executable(${test} ${test_srcs} ${test_knl})
    else()
      add_executable(${test} ${test_srcs})
    endif()
    TARGET_LINK_LIBRARIES(${test}
        MetaObject
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
        ${link_libs}
        MetaParameters)

    if(OpenCV_FOUND)
      TARGET_LINK_LIBRARIES(${test} ${OpenCV_LIBS})
    endif()
    ADD_DEPENDENCIES(${test} MetaObject)
    set_target_properties(${test} PROPERTIES FOLDER Tests/MetaObject)
    add_test(NAME ${test} COMMAND $<TARGET_FILE:${test}>)
    if(MSVC)
      CONFIGURE_FILE("tests/Test.vcxproj.user.in" ${CMAKE_BINARY_DIR}/${test}.vcxproj.user @ONLY)
      CONFIGURE_FILE("tests/Test.vcxproj.user.in" ${CMAKE_CURRENT_BINARY_DIR}/${test}.vcxproj.user @ONLY)
    endif()
  endforeach()
endif()
message(STATUS
    "-- MetaObject config\n"
	"  C++ flags \t${CMAKE_CXX_FLAGS}\n"
    "  C++ Rel flags\t${CMAKE_CXX_FLAGS_RELEASE}\n"
    "  C++ Deb flags\t${CMAKE_CXX_FLAGS_DEBUG}\n"
    "  OpenCV\t\t${OpenCV_VERSION} - ${OpenCV_DIR}\n"
    "  Qt5\t\t${Qt5_VERSION}\n"
    "  Python\t\t${HAVE_PYTHON}\n"
    "  Wt\t\t${Wt_FOUND}\n"
    "  ZeroMQ\t\t${ZeroMQ_FOUND}\n"
    "  Tests\t\t${Boost_UNIT_TEST_FRAMEWORK_FOUND}")
# ----------------------- examples ---------------------------
if(BUILD_EXAMPLES)
    ADD_SUBDIRECTORY("examples")
endif()
# ----------------------- install ----------------------------
install(TARGETS MetaObject MetaParameters
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
)


install(DIRECTORY include/
        DESTINATION include
        FILES_MATCHING PATTERN "*.hpp"
)

