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
    # The OpenCV instantiations only compile when MO_HAVE_OPENCV is set, so the
    # include path and libraries must be wired up here (other modules pick them
    # up from the environment, which is not portable).
    target_include_directories(metaobject_metaparams SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(metaobject_metaparams PUBLIC ${OpenCV_LIBRARIES})
endif()

IF(TARGET metaobject_qtgui)
  TARGET_LINK_LIBRARIES(metaobject_metaparams PUBLIC metaobject_qtgui)
ENDIF()

if(TARGET metaobject_wtgui)
  TARGET_LINK_LIBRARIES(metaobject_metaparams PUBLIC metaobject_wtgui)
endif()

# Guarded like the gui links above: metaobject_python is defined later in the
# build order (modules/python after modules/metaparams), so an unguarded
# reference here degrades to a raw -lmetaobject_python that cannot resolve.
# metaparams itself only uses boost::python directly (its undefined
# boost::python symbols resolve at runtime in a process hosting the python
# extension module, the same way the gui links work).
if(TARGET metaobject_python)
    TARGET_LINK_LIBRARIES(metaobject_metaparams PUBLIC metaobject_python)
endif()
