file(GLOB_RECURSE src "src/*.cpp" "src/*.hpp" "src/*.h" "src/*.cu")
metaobject_declare_module(NAME object
    SRC src
    DEPENDS
        metaobject_core
        metaobject_params
        RuntimeObjectSystem
        RuntimeCompiler
        Boost::filesystem
)


if(BUILD_TESTS)
    file(GLOB_RECURSE plugin_src "tests/plugin/*.cpp")
    add_library(mo_objectplugin SHARED ${plugin_src})
    target_link_libraries(mo_objectplugin metaobject_object)
    target_include_directories(mo_objectplugin
        PUBLIC ${CMAKE_CURRENT_LIST_DIR}/tests/plugin
    )

    file(GLOB_RECURSE tests "tests/unit_tests/*.cpp")
    add_executable(test_mo_object ${tests})
    target_link_libraries(test_mo_object metaobject_object metaobject_serialization mo_objectplugin)
    set_property(TARGET test_mo_object PROPERTY CXX_STANDARD 11)
    add_test(NAME test_mo_object COMMAND test_mo_object)
endif()
