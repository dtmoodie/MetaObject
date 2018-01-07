if(HAVE_PYTHON)
    metaobject_declare_module(NAME python DEPENDS metaobject_core metaobject_object)
    target_include_directories(metaobject_python PUBLIC $<BUILD_INTERFACE:${PYTHON_INCLUDE_DIR}>)
    target_link_libraries(metaobject_python ${Boost_PYTHON_LIBRARY_RELEASE} ${PYTHON_LIBRARY})

    if(UNIX)
        get_target_property(out_dir metaobject_python ARCHIVE_OUTPUT_DIRECTORY)
        configure_file(${CMAKE_CURRENT_LIST_DIR}/metaobject.py.in.unix ${out_dir}/metaobject.py @ONLY)
        add_custom_target(metaobject.py.in.unix ALL)
    else()
        get_target_property(out_dir metaobject_python ARCHIVE_OUTPUT_DIRECTORY)
        configure_file(${CMAKE_CURRENT_LIST_DIR}/metaobject.py.in.win32 ${out_dir}/Debug/metaobject.py @ONLY)
        configure_file(${CMAKE_CURRENT_LIST_DIR}/metaobject.py.in.win32 ${out_dir}/RelWithDebInfo/metaobject.py @ONLY)
        configure_file(${CMAKE_CURRENT_LIST_DIR}/metaobject.py.in.win32 ${out_dir}/Release/metaobject.py @ONLY)
    endif(UNIX)
    set(metaobject_param_include
        "${metaobject_param_include}\n#include <MetaObject/python/PythonPolicy.hpp>" CACHE INTERNAL "" FORCE)

endif()
