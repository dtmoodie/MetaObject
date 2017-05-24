metaobject_declare_module(NAME serialization DEPENDS metaobject_core metaobject_params metaobject_object)

target_include_directories(metaobject_serialization
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../dependencies/cereal/include>
)
