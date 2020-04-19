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
    target_link_libraries(test_mo_params metaobject_params metaobject_serialization gtest gtest_main)
    add_test(NAME test_mo_cuda COMMAND test_mo_params)

    
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        set_target_properties(test_mo_params PROPERTIES COMPILE_FLAGS "-fno-omit-frame-pointer -fsanitize=address")
        set_target_properties(test_mo_params PROPERTIES LINK_FLAGS "-fno-omit-frame-pointer -fsanitize=address")
    endif()
    
    set_target_properties(test_mo_params PROPERTIES FOLDER Tests/MetaObject/params)
    setup_msvc_target(test_mo_params)
endif()
