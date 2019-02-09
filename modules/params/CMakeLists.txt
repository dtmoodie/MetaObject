file(GLOB_RECURSE src "src/*.cpp" "src/*.hpp" "src/*.h" "src/*.cu")
metaobject_declare_module(NAME params
    SRC src
    DEPENDS
        metaobject_core
        ct
        metaobject_types
        metaobject_runtime_reflection
        cereal
)

target_include_directories(metaobject_params
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
)

if(BUILD_TESTS)
    file(GLOB_RECURSE tests "tests/*.cpp")
    add_executable(test_mo_params ${tests})
    target_link_libraries(test_mo_params metaobject_params)
    add_test(NAME test_mo_cuda COMMAND test_mo_params)
endif()
