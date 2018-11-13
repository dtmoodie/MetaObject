FILE(GLOB_RECURSE src "src/*.hpp" "src/*.cu" "src/*.cpp")

metaobject_declare_module(NAME cuda SRC src DEPENDS
    metaobject_core
)

if(BUILD_TESTS)
    file(GLOB_RECURSE tests "tests/*.cpp")
    add_executable(mo_test_cuda ${tests})
    target_link_libraries(mo_test_cuda metaobject_cuda)
    set_property(TARGET mo_test_cuda PROPERTY CXX_STANDARD 11)
endif()
