FILE(GLOB_RECURSE src "src/*.hpp" "src/*.cu" "src/*.cpp")
metaobject_declare_module(NAME types SRC src DEPENDS ct metaobject_core)

if(BUILD_TESTS)
    add_executable(test_mo_small_vec tests/test_mo_small_vec.cpp)
    target_link_libraries(test_mo_small_vec metaobject_types)
    set_target_properties(test_mo_small_vec PROPERTIES FOLDER Tests/MetaObject)
endif()
