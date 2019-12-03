file(GLOB_RECURSE src "src/*.cpp" "src/*.hpp" "src/*.h" "src/*.cu")
metaobject_declare_module(NAME metaparams
    SRC src
    DEPENDS 
      metaobject_params 
      metaobject_serialization 
      metaobject_types
)

if(OpenCV_FOUND)
    target_compile_definitions(metaobject_metaparams
        PUBLIC MO_HAVE_OPENCV
    )
endif()

IF(TARGET metaobject_qtgui)
  TARGET_LINK_LIBRARIES(metaobject_metaparams metaobject_qtgui)
ENDIF()

if(TARGET metaobject_wtgui)
  TARGET_LINK_LIBRARIES(metaobject_metaparams metaobject_wtgui)
endif()

if(MO_HAVE_PYTHON)
    TARGET_LINK_LIBRARIES(metaobject_metaparams metaobject_python)
endif()
