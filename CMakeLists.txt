cmake_minimum_required(VERSION 2.8)
CMAKE_POLICY(SET CMP0048 NEW)
PROJECT(MetaObject
    VERSION 0.1)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
CMAKE_POLICY(SET CMP0020 NEW)
include("${CMAKE_CURRENT_LIST_DIR}/cmake/CMakeMacros.cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(link_libs "")

option(BUILD_TESTS "Build tests" ON)
option(BUILD_DEPENDENCIES "Build dependencies" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_DEPENDENCIES "Build dependencies within the AquilOS family" ON)
option(WITH_QT "Enable QT support" ON)
option(WITH_PYTHON "Enable Python bindings" ON)
option(WITH_OPENCV "Enable OpenCV types" ON )
option(WITH_CUDA "Enable CUDA recompilation" ON)
option(WITH_WT "Enable WT support" ON)
option(WITH_ZEROMQ "Enable ZEROMQ support" OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include(cmake/OpenCVPCHSupport.cmake)

set(link_dirs_deb "")
set(link_dirs_rel "")
set(compile_flags "")

set(HAVE_PYTHON OFF)

include(cmake/metaobject_dependencies.cmake)

if(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /FC /Zi /MP /Zm512 /bigobj /FS")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zo /d2Zi+ /Oy-")
  ADD_DEFINITIONS(-D_ENABLE_ATOMIC_ALIGNMENT_FIX)
else(WIN32)
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

# -------------------- Profiling ---------------------------------
set(MO_PROFILING_MODE "BUILTIN" CACHE STRING "Set MetaObject profiling mode")
set_property(CACHE MO_PROFILING_MODE PROPERTY STRINGS BUILTIN DYNAMIC NONE)

find_package(NVToolsExt QUIET)
if(NVToolsExt)
    if(NV_TOOLS_EXT_INCLUDE_DIR)
        include_directories(${NV_TOOLS_EXT_INCLUDE_DIR})
    endif(NV_TOOLS_EXT_INCLUDE_DIR)
    add_definitions(-DHAVE_NVTX)
endif()
if(${MO_PROFILING_MODE} STREQUAL "BUILTIN")
  add_definitions(-DRMT_BUILTIN)
endif()
if(${MO_PROFILING_MODE} STREQUAL "NONE")
  add_definitions(-DPROFILING_NONE)
endif()


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

include(cmake/metaobject_declare_module.cmake)

WRITE_RCC_CONFIG("${target_include_dirs_}" "${LINK_DIRS_DEBUG}" "${LINK_DIRS_RELEASE}" "")

# ---------------------- modules ------------------------------------
set(metaobject_modules "" CACHE INTERNAL "" FORCE)
macro(GroupSources basedir curdir tab)
file(GLOB children RELATIVE "${basedir}/${curdir}" "${basedir}/${curdir}/*")
    foreach(child ${children})
        if(IS_DIRECTORY ${basedir}/${curdir}/${child})
            GroupSources("${basedir}" ${curdir}/${child} "${tab}  ")
        else()
            string(REPLACE "/" "\\" groupname ${curdir})
            source_group(${groupname} FILES ${basedir}/${curdir}/${child})
        endif()
   endforeach()
endmacro()
set(metaobject_module_includes "" CACHE INTERNAL "" FORCE)


SUBDIRLIST(modules "${CMAKE_CURRENT_LIST_DIR}/modules")
foreach( module ${modules})
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/modules/${module})
endforeach()
set(MetaObject_LIBRARIES "${metaobject_modules}" CACHE STRING "" FORCE)
set(MetaObject_INCLUDE_DIRS "${metaobject_module_includes};${cereal_INCLUDE_DIRS};${RCC_INCLUDE_DIRS}" CACHE PATH "" FORCE)
# -------------------- tests ----------------------------------
if(BUILD_TESTS)
    add_subdirectory(tests)
endif(BUILD_TESTS)

# ---------------------- status message -----------------------------
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
    "  Tests\t\t${Boost_UNIT_TEST_FRAMEWORK_FOUND}"
)

# ----------------------- examples ---------------------------
if(BUILD_EXAMPLES)
    ADD_SUBDIRECTORY("examples")
endif()
# ----------------------- install ----------------------------
file(COPY "cmake" DESTINATION ${PROJECT_BINARY_DIR})

export(PACKAGE MetaObject)

if(Qt5_FOUND)
    set(MO_QT5 ON)
else(Qt5_FOUND)
    set(MO_QT5 OFF)
endif(Qt5_FOUND)

if(Wt_FOUND)
    set(MO_WT ON)
else(Wt_FOUND)
    set(MO_WT OFF)
endif(Wt_FOUND)

install(DIRECTORY dependencies/cereal/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

install(DIRECTORY dependencies/cereal/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

if(WIN32)
    install(DIRECTORY ${CMAKE_BINARY_DIR}/Debug/
        DESTINATION bin
        FILES_MATCHING PATTERN "*d.pdb"
    )

    install(DIRECTORY ${CMAKE_BINARY_DIR}/RelWithDebInfo/
        DESTINATION bin
        FILES_MATCHING PATTERN "*.pdb"
    )

    install(DIRECTORY ${CMAKE_BINARY_DIR}/Debug/
        DESTINATION lib
        FILES_MATCHING PATTERN "*d.pdb"
    )

    install(DIRECTORY ${CMAKE_BINARY_DIR}/RelWithDebInfo/
        DESTINATION lib
        FILES_MATCHING PATTERN "*.pdb"
    )

endif(WIN32)

install(DIRECTORY cmake DESTINATION share/MetaObject FILES_MATCHING PATTERN "*.cmake")

install(FILES
    "${PROJECT_BINARY_DIR}/MetaObjectConfig.cmake"
    DESTINATION "${CMAKE_INSTALL_PREFIX}/share/MetaObject" COMPONENT dev
)
