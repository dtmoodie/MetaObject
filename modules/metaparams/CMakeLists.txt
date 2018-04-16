file(GLOB_RECURSE src "src/*.cpp" "src/*.hpp" "src/*.h" "src/*.cu")
metaobject_declare_module(NAME metaparams SRC src DEPENDS metaobject_params metaobject_serialization metaobject_types)
IF(TARGET metaobject_qtgui)
  TARGET_LINK_LIBRARIES(metaobject_metaparams metaobject_qtgui)
ENDIF()

if(TARGET metaobject_wtgui)
  TARGET_LINK_LIBRARIES(metaobject_metaparams metaobject_wtgui)
endif()

if(HAVE_PYTHON)
    TARGET_LINK_LIBRARIES(metaobject_metaparams metaobject_python)
endif()
