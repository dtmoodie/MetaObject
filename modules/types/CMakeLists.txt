FILE(GLOB_RECURSE src "src/*.hpp" "src/*.cu" "src/*.cpp")
metaobject_declare_module(NAME types SRC src DEPENDS ct metaobject_core ${Boost_FILESYSTEM_LIBRARY_RELEASE} ${Boost_FILESYSTEM_LIBRARY_DEBUG})

if(BUILD_TESTS)
    add_executable(test_mo_types tests/test_mo_small_vec.cpp tests/main.cpp)
    target_link_libraries(test_mo_types metaobject_types gtest gtest_main)
    set_target_properties(test_mo_types PROPERTIES FOLDER Tests/MetaObject/types)
    setup_msvc_target(test_mo_types)
endif()
