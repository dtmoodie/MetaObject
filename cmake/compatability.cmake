find_package(CUDA QUIET)
if(CUDA_FOUND AND CUDA_VERSION_MAJOR STREQUAL "6" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 5.0)
    set(JETSON_COMPATABILITY ON)
else()
    set(JETSON_COMPATABILITY OFF)
endif()
