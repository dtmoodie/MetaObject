metaobject_declare_module(NAME visitation
    DEPENDS
        metaobject_core
        metaobject_types
)

if(BUILD_TESTS)
    file(GLOB src "tests/*.cpp" "tests/*.hpp")
    add_executable(test_mo_visitation ${src})
    target_link_libraries(test_mo_visitation metaobject_visitation)
    set_target_properties(test_mo_visitation PROPERTIES FOLDER Tests/metaobject/visitation)

    if(OpenCV_FOUND)
        target_compile_definitions(test_mo_visitation
            PRIVATE HAVE_OPENCV
        )
        target_link_libraries(test_mo_visitation ${OpenCV_LIBRARIES})
    endif()
endif()
