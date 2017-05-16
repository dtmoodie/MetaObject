if(Qt5_FOUND)
    include_directories(${Qt5Core_INCLUDE_DIRS};${Qt5Widgets_INCLUDE_DIRS})
    file(GLOB_RECURSE hdr "src/*.hpp")
    file(GLOB_RECURSE src "src/*.cpp")
    QT5_WRAP_CPP(MOC "src/MetaObject/params/ui/Qt/SignalProxy.hpp")
    set(src "${src};${MOC};${hdr}")
    metaobject_declare_module(NAME gui SRC src DEPENDS ${OpenCV_LIBS}
        Qt5::Core Qt5::Widgets metaobject_core metaobject_params
    )
endif()
