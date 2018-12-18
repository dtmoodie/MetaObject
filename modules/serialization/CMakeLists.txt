metaobject_declare_module(NAME serialization
    DEPENDS metaobject_core metaobject_params metaobject_types cereal
)


if(BUILD_TESTS)
    file(GLOB src "tests/*.cpp" "tests/*.hpp")
    add_executable(test_mo_serialization ${src})
    target_link_libraries(test_mo_serialization metaobject_serialization)
    set_target_properties(test_mo_serialization PROPERTIES FOLDER Tests/metaobject/serialization)
endif()
