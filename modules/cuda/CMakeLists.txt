FILE(GLOB_RECURSE src "src/*.hpp" "src/*.cu" "src/*.cpp")

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

metaobject_declare_module(NAME cuda SRC src DEPENDS
    metaobject_core
)

if(BUILD_TESTS)
    find_package(OpenCV QUIET COMPONENTS cudaarithm)

    file(GLOB_RECURSE tests "tests/*.cpp" "tests/*.cu")
    cuda_add_executable(test_mo_cuda ${tests})
    target_include_directories(test_mo_cuda
        PRIVATE ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(test_mo_cuda metaobject_cuda ${OpenCV_LIBRARIES})
    set_property(TARGET test_mo_cuda PROPERTY CXX_STANDARD 11)

    INCLUDE_DIRECTORIES("../core/src")
    INCLUDE_DIRECTORIES("../core/jetson_patch")
    cuda_add_executable(test_cuda_fiber tests2/cuda_build.cu)
    target_link_libraries(test_cuda_fiber Boost::unit_test_framework)
endif()
