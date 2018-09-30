file(GLOB hdr "*.hpp")

add_executable(test_mo_dynamic_access ${hdr} print.cpp)
target_link_libraries(test_mo_dynamic_access metaobject_serialization)
set_target_properties(test_mo_dynamic_access PROPERTIES FOLDER Tests/metaobject/serialization)
if(TARGET cereal)
    target_link_libraries(test_mo_dynamic_access cereal)
endif()
