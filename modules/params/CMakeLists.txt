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
    add_test(NAME test_mo_params COMMAND test_mo_params)

    
    # libasan 9.x/10.x crash at startup on modern host kernels: during
    # _dl_init, AsanInitInternal lazily binds pthread_getspecific via
    # _dl_fixup while the lookup itself faults (SEGV before the first test;
    # a trivial ASan hello-world reproduces it under focal's g++ 9.4, and
    # test_mo_params crashes with the same signature under g++-10, so this is
    # a libasan runtime bug on the host, not a code bug). The g++-11+ cells
    # (libasan 11+, clean on the same host) keep the full ASan coverage of
    # this binary.
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" AND
        NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
        set_target_properties(test_mo_params PROPERTIES COMPILE_FLAGS "-fno-omit-frame-pointer -fsanitize=address")
        set_target_properties(test_mo_params PROPERTIES LINK_FLAGS "-fno-omit-frame-pointer -fsanitize=address")
    endif()
    
    set_target_properties(test_mo_params PROPERTIES FOLDER Tests/MetaObject/params)
    setup_msvc_target(test_mo_params)
endif()
