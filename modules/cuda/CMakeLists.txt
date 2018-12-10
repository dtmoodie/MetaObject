FILE(GLOB_RECURSE src "src/*.hpp" "src/*.cu" "src/*.cpp")

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

metaobject_declare_module(NAME cuda SRC src DEPENDS
    metaobject_core
)

if(BUILD_TESTS)
    file(GLOB_RECURSE tests "tests/*.cpp" "tests/*.cu")
    cuda_add_executable(test_mo_cuda ${tests})
    target_link_libraries(test_mo_cuda metaobject_cuda)
    set_property(TARGET test_mo_cuda PROPERTY CXX_STANDARD 11)
endif()
