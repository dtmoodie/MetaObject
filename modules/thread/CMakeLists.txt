metaobject_declare_module(NAME thread DEPENDS metaobject_signals)
target_include_directories(metaobject_thread PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>)