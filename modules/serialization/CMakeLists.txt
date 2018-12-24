metaobject_declare_module(NAME serialization
    DEPENDS metaobject_core metaobject_params metaobject_types cereal
)


if(BUILD_TESTS)
    find_package(OpenCV QUIET)
    file(GLOB_RECURSE src "tests/*.cpp" "tests/*.hpp")
    add_executable(test_mo_serialization ${src})

    if(OpenCV_FOUND)
        target_compile_definitions(test_mo_serialization
            PRIVATE HAVE_OPENCV
        )
        target_link_libraries(test_mo_serialization ${OpenCV_LIBRARIES})
    endif()
    target_link_libraries(test_mo_serialization metaobject_serialization)
    set_target_properties(test_mo_serialization PROPERTIES FOLDER Tests/metaobject/serialization)
endif()
