if(WITH_PYTHON)
    find_package(PythonLibs 2.7 QUIET)
    find_package(PythonInterp 2.7 QUIET)
    if(PythonLibs_FOUND AND PythonInterp_FOUND)
        if(WIN32)
            find_package(Boost QUIET COMPONENTS python python2)
            if(Boost_PYTHON_FOUND)
                find_package(NumPy 1.7.1)
                if(NUMPY_FOUND)
                    set(Boost_PYTHON_LIBRARY_RELEASE "${Boost_PYTHON-PY${boost_py_version}_LIBRARY_RELEASE}" CACHE PATH "" FORCE)
                    set(Boost_PYTHON_LIBRARY_DEBUG "${Boost_PYTHON-PY${boost_py_version}_LIBRARY_DEBUG}" CACHE PATH "" FORCE)
                    set(MO_HAVE_PYTHON 1 CACHE BOOL INTERNAL FORCE)
                    set(MO_PYTHON_STATUS "${PYTHONLIBS_VERSION_STRING} include: ${PYTHON_INCLUDE_DIR} Boost: ${Boost_PYTHON_LIBRARY_RELEASE} Numpy: ${NUMPY_VERSION}" CACHE BOOL INTERNAL FORCE)
                    list(APPEND link_libs "${Boost_PYTHON_LIBRARY_RELEASE};${PYTHON_LIBRARY}")
                    get_filename_component(_python_lib_dirs "${PYTHON_LIBRARY}" DIRECTORY)
                    foreach(dir ${_python_lib_dirs})
                        if(NOT ${dir} STREQUAL debug AND NOT ${dir} STREQUAL optimized)
                            link_directories(${dir})
                        endif()
                    endforeach()
                    include_directories(${PYTHON_INCLUDE_DIR})
                    set(MO_HAVE_PYTHON 1 CACHE BOOL INTERNAL FORCE)
                else()
                    set(MO_PYTHON_STATUS "Unable to find numpy" CACHE BOOL INTERNAL FORCE)
                endif()                
            endif()
        else(WIN32)
            # Find the matching boost python implementation
            set(version ${PYTHONLIBS_VERSION_STRING})
            STRING( REGEX REPLACE "[^0-9]" "" boost_py_version ${version} )
            find_package(Boost 1.46 QUIET COMPONENTS "python${boost_py_version}")
            set(Boost_PYTHON_FOUND ${Boost_PYTHON-PY${boost_py_version}_FOUND})

            while(NOT "${version}" STREQUAL "" AND NOT Boost_PYTHON_FOUND)
                STRING( REGEX REPLACE "([0-9.]+).[0-9]+" "\\1" version ${version} )

                STRING( REGEX REPLACE "[^0-9]" "" boost_py_version ${version} )
                find_package(Boost 1.46 QUIET COMPONENTS "python${boost_py_version}")
                set(Boost_PYTHON_FOUND ${Boost_PYTHON${boost_py_version}_FOUND})

                STRING( REGEX MATCHALL "([0-9.]+).[0-9]+" has_more_version ${version} )
                if("${has_more_version}" STREQUAL "")
                  break()
                endif()
            endwhile()


            if(Boost_PYTHON_FOUND)
                find_package(NumPy 1.7.1)
                if(NUMPY_FOUND)
                    set(Boost_PYTHON_LIBRARY_RELEASE "${Boost_PYTHON${boost_py_version}_LIBRARY_RELEASE}" CACHE PATH "" FORCE)
                    set(Boost_PYTHON_LIBRARY_DEBUG "${Boost_PYTHON${boost_py_version}_LIBRARY_DEBUG}" CACHE PATH "" FORCE)
                    set(MO_HAVE_PYTHON 1 CACHE BOOL INTERNAL FORCE)
                    set(MO_PYTHON_STATUS "${PYTHONLIBS_VERSION_STRING} include: ${PYTHON_INCLUDE_DIR} Boost: ${Boost_PYTHON_LIBRARY_RELEASE} Numpy: ${NUMPY_VERSION}" CACHE BOOL INTERNAL FORCE)
                else()
                    set(MO_PYTHON_STATUS "Unable to find numpy" CACHE BOOL INTERNAL FORCE)
                endif()
            else()
                set(MO_PYTHON_STATUS "Unable to find a suitable version of boost python" CACHE BOOL INTERNAL FORCE)
            endif()
        endif(WIN32)
    else(PythonLibs_FOUND AND PythonInterp_FOUND)
        set(MO_PYTHON_STATUS "Unable to find PythonLibs and PythonInterp" CACHE BOOL INTERNAL FORCE)
    endif()

else(WITH_PYTHON)
    set(MO_PYTHON_STATUS "Python disabled by WITH_PYTHON cmake flag" CACHE BOOL INTERNAL FORCE)
endif()
