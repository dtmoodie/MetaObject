metaobject_declare_module(NAME logging DEPENDS metaobject_thread)
target_include_directories(metaobject_logging PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>)