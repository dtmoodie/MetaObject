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
    target_include_directories(metaobject_metaparams
      PUBLIC
        "${OpenCV_INCLUDE_DIRS}"
    )
endif()

IF(TARGET metaobject_qtgui)
  TARGET_LINK_LIBRARIES(metaobject_metaparams PUBLIC metaobject_qtgui)
ENDIF()

if(TARGET metaobject_wtgui)
  TARGET_LINK_LIBRARIES(metaobject_metaparams PUBLIC metaobject_wtgui)
endif()

if(MO_HAVE_PYTHON2)
    TARGET_LINK_LIBRARIES(metaobject_metaparams PUBLIC metaobject_python)
endif()
