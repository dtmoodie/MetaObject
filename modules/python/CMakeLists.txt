if(MO_HAVE_PYTHON AND MO_HAVE_OPENCV)
    LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
    metaobject_declare_module(NAME python
        DEPENDS
            metaobject_core
            metaobject_object
            metaobject_cuda
            ${OpenCV_LIBRARIES}
            ${PYTHON_LIBRARY}
            ct
        INCLUDES
            $<BUILD_INTERFACE:${PYTHON_INCLUDE_DIR}>
            $<BUILD_INTERFACE:${NUMPY_INCLUDE_DIR}>
    )
    if(TARGET Boost::python)
        target_link_libraries(metaobject_python PUBLIC Boost::python)
    endif()
    if(TARGET Boost::python27)
        target_link_libraries(metaobject_python PUBLIC Boost::python27)
    endif()
    if(Boost_PYTHON_LIBRARY_RELEASE)
        target_link_libraries(metaobject_python PUBLIC ${Boost_PYTHON_LIBRARY_RELEASE})
    endif()
    
    if(Boost_PYTHON_LIBRARY_DEBUG)
        target_link_libraries(metaobject_python PUBLIC ${Boost_PYTHON_LIBRARY_DEBUG})
    endif()

    if(UNIX)
        set(POSTFIX "")
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(POSTFIX "${CMAKE_DEBUG_POSTFIX}")
        endif()
        get_target_property(out_dir metaobject_python ARCHIVE_OUTPUT_DIRECTORY)
        configure_file(${CMAKE_CURRENT_LIST_DIR}/metaobject.py.in.unix ${out_dir}/metaobject.py @ONLY)
        add_custom_target(metaobject.py.in.unix ALL)
        configure_file(${CMAKE_CURRENT_LIST_DIR}/tests/test_python.py ${out_dir}/metaobject_python_test.py @ONLY)
    else()
        get_target_property(out_dir metaobject_python ARCHIVE_OUTPUT_DIRECTORY)
        configure_file(${CMAKE_CURRENT_LIST_DIR}/metaobject.py.in.win32 ${out_dir}/Debug/metaobject.py @ONLY)
        configure_file(${CMAKE_CURRENT_LIST_DIR}/metaobject.py.in.win32 ${out_dir}/RelWithDebInfo/metaobject.py @ONLY)
        configure_file(${CMAKE_CURRENT_LIST_DIR}/metaobject.py.in.win32 ${out_dir}/Release/metaobject.py @ONLY)
        configure_file(${CMAKE_CURRENT_LIST_DIR}/tests/test_python.py ${out_dir}/Debug/metaobject_python_test.py @ONLY)
        configure_file(${CMAKE_CURRENT_LIST_DIR}/tests/test_python.py ${out_dir}/RelWithDebInfo/metaobject_python_test.py @ONLY)
    endif(UNIX)
    set(metaobject_param_include
        "${metaobject_param_include}\n#ifdef slots\n#undef slots\n#endif\n#include <MetaObject/python/PythonPolicy.hpp>\n" CACHE INTERNAL "" FORCE
    )

    if(BUILD_TESTS)
        add_executable(test_mo_python_init tests/python_init_param_setter.cpp)
        set_target_properties(test_mo_python_init PROPERTIES FOLDER Tests/MetaObject/python)
        target_link_libraries(test_mo_python_init 
            ct 
            metaobject_python 
            metaobject_object
            ${PYTHON_LIBRARY}
        )

        add_test(NAME test_mo_python COMMAND python2 test_mo_python.py)
    endif()
endif()
