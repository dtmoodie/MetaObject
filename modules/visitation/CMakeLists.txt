metaobject_declare_module(NAME visitation
    DEPENDS
        metaobject_core
        metaobject_types
)

if(BUILD_TESTS)
    #file(GLOB src "tests/*.cpp" "tests/*.hpp")
    #add_executable(test_mo_visitation ${src})
    #target_link_libraries(test_mo_visitation metaobject_serialization)
    #set_target_properties(test_mo_visitation PROPERTIES FOLDER Tests/metaobject/serialization)
endif()
