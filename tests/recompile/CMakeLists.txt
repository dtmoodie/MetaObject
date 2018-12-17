set(CUDA_NVCC_FLAGS "-std=c++11")
if(NOT JETSON_COMPATABILITY)
    cuda_add_library(test_recompile_object SHARED obj.cpp)
else()
    add_library(test_recompile_object SHARED obj.cpp)
endif()

TARGET_LINK_LIBRARIES(test_recompile_object
    ${MetaObject_LIBRARIES}
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    metaobject_params
)
RCC_TARGET_CONFIG(test_recompile_object deb rel)

add_executable(test_recompile test_recompile.cpp)
add_dependencies(test_recompile test_recompile_object)
TARGET_LINK_LIBRARIES(test_recompile
    ${MetaObject_LIBRARIES}
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    metaobject_params
    test_recompile_object
)

set_target_properties(test_recompile_object PROPERTIES FOLDER Tests/MetaObject)
set_target_properties(test_recompile PROPERTIES FOLDER Tests/MetaObject)
