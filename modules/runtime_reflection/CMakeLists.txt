metaobject_declare_module(NAME runtime_reflection
    DEPENDS
        metaobject_core
        metaobject_types
)

if(BUILD_TESTS)
    file(GLOB src "tests/*.cpp" "tests/*.hpp")
    add_executable(test_mo_runtime_reflection ${src})
    target_link_libraries(test_mo_runtime_reflection metaobject_runtime_reflection)
    set_target_properties(test_mo_runtime_reflection PROPERTIES FOLDER Tests/metaobject)

    if(OpenCV_FOUND)
        target_compile_definitions(test_mo_runtime_reflection
            PRIVATE MO_HAVE_OPENCV
        )
        target_link_libraries(test_mo_runtime_reflection ${OpenCV_LIBRARIES})
    endif()
    add_test(NAME test_mo_runtime_reflection COMMAND test_mo_runtime_reflection)
endif()
