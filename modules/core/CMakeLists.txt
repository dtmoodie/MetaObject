FILE(GLOB_RECURSE src "src/*.hpp" "src/*.cu" "src/*.cpp")

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

metaobject_declare_module(NAME core SRC src DEPENDS 
    ${OpenCV_LIBS} 
    ${Boost_TARGETS} 
    cereal 
    RuntimeObjectSystem
)
target_include_directories(metaobject_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
)
if(${CUDA_FOUND})
    target_link_libraries(metaobject_core ${CUDA_CUDART_LIBRARY})
endif(${CUDA_FOUND})
