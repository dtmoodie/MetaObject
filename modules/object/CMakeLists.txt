file(GLOB_RECURSE src "src/*.cpp" "src/*.hpp" "src/*.h" "src/*.cu")
metaobject_declare_module(NAME object SRC src DEPENDS metaobject_core metaobject_params RuntimeObjectSystem RuntimeCompiler Boost::filesystem)


if(BUILD_TESTS)
    file(GLOB_RECURSE tests "tests/*.cpp")
    add_executable(test_mo_object ${tests})
    target_link_libraries(test_mo_object metaobject_object)
    set_property(TARGET test_mo_object PROPERTY CXX_STANDARD 11)
endif()
