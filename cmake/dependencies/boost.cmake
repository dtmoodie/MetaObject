set(Boost_required_components system thread fiber filesystem)
if(BUILD_TESTS)
    list(APPEND Boost_required_components unit_test_framework)
endif()
set(Boost_USE_STATIC_LIBS        OFF)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME     OFF)
ADD_DEFINITIONS(-DBOOST_ALL_DYN_LINK)
find_package(Boost REQUIRED COMPONENTS ${Boost_required_components})
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIR_DEBUG})
link_directories(${Boost_LIBRARY_DIR})
list(APPEND PROJECT_BIN_DIRS_DEBUG_ ${Boost_LIBRARY_DIR})
list(APPEND PROJECT_BIN_DIRS_RELEASE_ ${Boost_LIBRARY_DIR})
list(APPEND PROJECT_BIN_DIRS_RELWITHDEBINFO_ ${Boost_LIBRARY_DIR})
list(APPEND PROJECT_BIN_DIRS_DEBUG_ ${Boost_LIBRARY_DIR_DEBUG})
list(APPEND PROJECT_BIN_DIRS_RELEASE_ ${Boost_LIBRARY_DIR_RELEASE})
list(APPEND PROJECT_BIN_DIRS_RELWITHDEBINFO_ ${Boost_LIBRARY_DIR_RELEASE})
set(bin_dirs_ "${BIN_DIRS};Boost")
list(REMOVE_DUPLICATES bin_dirs_)
set(BIN_DIRS "${bin_dirs_}" CACHE STRING "" FORCE)
set(Boost_TARGETS "")
foreach(cmp ${Boost_required_components})
if(TARGET Boost::${cmp})
    list(APPEND Boost_TARGETS "Boost::${cmp}")
endif()
endforeach()
