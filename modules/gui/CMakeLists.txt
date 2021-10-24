if(MO_HAVE_QT)
    find_package(OpenCV QUIET)
    if(OpenCV_FOUND)
        file(GLOB_RECURSE hdr "src/*.hpp")
        file(GLOB_RECURSE src "src/MetaObject/params/ui/Qt/*.cpp")
        list(APPEND src "src/MetaObject/params/ui/Qt.cpp")
        QT5_WRAP_CPP(MOC "src/MetaObject/params/ui/Qt/SignalProxy.hpp")
        set(src "${src};${MOC};${hdr}")
        metaobject_declare_module(NAME qtgui SRC src DEPENDS ${OpenCV_LIBS}
            Qt5::Core 
            Qt5::Widgets 
            metaobject_core 
            metaobject_params 
            metaobject_types
        )
        target_include_directories(metaobject_qtgui 
            PUBLIC
                $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>
        )
    endif()
endif(MO_HAVE_QT)

if(MO_HAVE_WT)
    file(GLOB_RECURSE hdr "src/*.hpp")
    file(GLOB_RECURSE src "src/MetaObject/params/ui/Wt/*.cpp")
    list(APPEND src "src/MetaObject/params/ui/WT.cpp")

    set(src "${src};${hdr}")
    metaobject_declare_module(NAME wtgui SRC src DEPENDS ${OpenCV_LIBS}
        metaobject_core metaobject_params
    )
    target_include_directories(metaobject_wtgui PUBLIC $<BUILD_INTERFACE:${Wt_INCLUDE_DIR}>)
    target_link_libraries(metaobject_wtgui
        debug;${Wt_HTTP_DEBUG_LIBRARY}
        debug;${Wt_DEBUG_LIBRARY}
        optimized;${Wt_LIBRARY}
        optimized;${Wt_HTTP_LIBRARY}
        metaobject_types
    )
endif(MO_HAVE_WT)
