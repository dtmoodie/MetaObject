metaobject_declare_module(NAME serialization
    DEPENDS metaobject_core metaobject_params metaobject_types cereal
)

if(BUILD_TESTS)
    if(WITH_OPENCV)
        find_package(OpenCV QUIET)
    endif()
    file(GLOB_RECURSE src "tests/*.cpp" "tests/*.hpp")
    add_executable(test_mo_serialization ${src})
    target_include_directories(test_mo_serialization
        PRIVATE
            "${CMAKE_CURRENT_LIST_DIR}/../runtime_reflection/tests"
    )

    if(OpenCV_FOUND)
        target_compile_definitions(test_mo_serialization
            PRIVATE MO_HAVE_OPENCV
        )
        target_link_libraries(test_mo_serialization ${OpenCV_LIBRARIES})
    endif()
    target_link_libraries(test_mo_serialization metaobject_serialization gtest gtest_main)
    set_target_properties(test_mo_serialization PROPERTIES FOLDER Tests/MetaObject/serialization)
    add_test(NAME test_mo_serialization COMMAND test_mo_serialization)
    setup_msvc_target(test_mo_serialization)
endif()
