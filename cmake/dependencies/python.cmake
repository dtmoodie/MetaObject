if(WITH_PYTHON)
    # CMake 3.27 removed the legacy FindPythonInterp/FindPythonLibs modules:
    # on newer CMake (noble and the cuda12 image ship 3.28) the calls fail
    # silently and the python module never builds, so use the modern
    # FindPython module there and replicate the legacy variables the rest of
    # this file expects. The legacy path remains for the older CMake the
    # old-toolchain support targets (cmake_minimum_required is 2.8).
    if(CMAKE_VERSION VERSION_LESS 3.12)
        find_package(PythonLibs 2.7 QUIET)
        find_package(PythonInterp 2.7 QUIET)
    else()
        find_package(Python COMPONENTS Interpreter Development NumPy)
        if(Python_FOUND)
            # Replicate the legacy variables, in BOTH casings: find_package
            # sets <PackageName>_FOUND (PythonLibs_FOUND), while the legacy
            # FindPythonLibs/FindPythonInterp modules also exported the all
            # caps PYTHONLIBS_FOUND/PYTHONINTERP_FOUND, which the vendored
            # FindNumPy checks. CMake variables are case-sensitive, so both
            # spellings are needed downstream.
            set(PythonLibs_FOUND TRUE)
            set(PYTHONLIBS_FOUND TRUE)
            set(PythonInterp_FOUND TRUE)
            set(PYTHONINTERP_FOUND TRUE)
            set(PYTHON_EXECUTABLE "${Python_EXECUTABLE}")
            # the modern FindPython module sets the plural Python_INCLUDE_DIRS
            set(PYTHON_INCLUDE_DIR "${Python_INCLUDE_DIRS}")
            set(PYTHON_LIBRARY "${Python_LIBRARIES}")
            set(PYTHONLIBS_VERSION_STRING "${Python_VERSION}")
        endif()
    endif()
    if(PythonLibs_FOUND AND PythonInterp_FOUND)
        if(WIN32)
            find_package(Boost QUIET COMPONENTS python python2)
            if(Boost_PYTHON_FOUND)
                find_package(NumPy 1.7.1)
                if(NUMPY_FOUND)
                    set(Boost_PYTHON_LIBRARY_RELEASE "${Boost_PYTHON-PY${boost_py_version}_LIBRARY_RELEASE}" CACHE PATH "" FORCE)
                    set(Boost_PYTHON_LIBRARY_DEBUG "${Boost_PYTHON-PY${boost_py_version}_LIBRARY_DEBUG}" CACHE PATH "" FORCE)
                    set(MO_HAVE_PYTHON 1 CACHE BOOL INTERNAL FORCE)
                    set(MO_PYTHON_STATUS "${PYTHONLIBS_VERSION_STRING} include: ${PYTHON_INCLUDE_DIR} Boost: ${Boost_PYTHON_LIBRARY_RELEASE} Numpy: ${NUMPY_VERSION}")
                    list(APPEND link_libs "${Boost_PYTHON_LIBRARY_RELEASE};${PYTHON_LIBRARY}")
                    get_filename_component(_python_lib_dirs "${PYTHON_LIBRARY}" DIRECTORY)
                    foreach(dir ${_python_lib_dirs})
                        if(NOT ${dir} STREQUAL debug AND NOT ${dir} STREQUAL optimized)
                            link_directories(${dir})
                        endif()
                    endforeach()
                    set(MO_HAVE_PYTHON 1 CACHE BOOL INTERNAL FORCE)
                else()
                    set(MO_PYTHON_STATUS "Unable to find numpy" CACHE STRING INTERNAL FORCE)
                endif()
            endif()
        else(WIN32)
            # Find the matching boost python implementation
            set(searched_libs "")
            set(version ${PYTHONLIBS_VERSION_STRING})
            STRING( REGEX REPLACE "[^0-9]" "" boost_py_version ${version} )
            find_package(Boost 1.46 QUIET COMPONENTS "python-py${boost_py_version}")
            set(Boost_PYTHON_FOUND ${Boost_PYTHON-PY${boost_py_version}_FOUND})

            # older versions of boost call it boost-py
            while(NOT "${version}" STREQUAL "" AND NOT Boost_PYTHON_FOUND)
                STRING( REGEX REPLACE "([0-9.]+).[0-9]+" "\\1" version ${version} )

                STRING( REGEX REPLACE "[^0-9]" "" boost_py_version ${version} )
                find_package(Boost 1.46 QUIET COMPONENTS "python-py${boost_py_version}")
                list(APPEND searched_libs "python-py${boost_py_version}")
                set(Boost_PYTHON_FOUND ${Boost_PYTHON-PY${boost_py_version}_FOUND})

                STRING( REGEX MATCHALL "([0-9.]+).[0-9]+" has_more_version ${version} )
                if("${has_more_version}" STREQUAL "" OR Boost_PYTHON_FOUND)
                  set(Boost_PYTHON_LIBRARY_RELEASE "${Boost_PYTHON-PY${boost_py_version}_LIBRARY_RELEASE}" CACHE PATH "" FORCE)
                  set(Boost_PYTHON_LIBRARY_DEBUG "${Boost_PYTHON-PY${boost_py_version}_LIBRARY_DEBUG}" CACHE PATH "" FORCE)
                  break()
                endif()
            endwhile()

            # newever versions of boost call it boost-python
            set(version ${PYTHONLIBS_VERSION_STRING})
            if(NOT Boost_PYTHON_FOUND)
                while(NOT "${version}" STREQUAL "" AND NOT Boost_PYTHON_FOUND)
                    STRING( REGEX REPLACE "([0-9.]+).[0-9]+" "\\1" version ${version} )

                    STRING( REGEX REPLACE "[^0-9]" "" boost_py_version ${version} )
                    find_package(Boost 1.46 QUIET COMPONENTS "python${boost_py_version}")
                    list(APPEND searched_libs "python${boost_py_version}")
                    set(Boost_PYTHON_FOUND ${Boost_PYTHON${boost_py_version}_FOUND})

                    STRING( REGEX MATCHALL "([0-9.]+).[0-9]+" has_more_version ${version} )
                    if("${has_more_version}" STREQUAL "" OR Boost_PYTHON_FOUND)
                        set(Boost_PYTHON_LIBRARY_RELEASE "${Boost_PYTHON${boost_py_version}_LIBRARY_RELEASE}" CACHE PATH "" FORCE)
                        set(Boost_PYTHON_LIBRARY_DEBUG "${Boost_PYTHON${boost_py_version}_LIBRARY_DEBUG}" CACHE PATH "" FORCE)
                      break()
                    endif()
                endwhile()
            endif()


            # Config-mode Boost (Debian/Ubuntu ship BoostConfig.cmake, which
            # the vendored FindBoost delegates to) sets none of the legacy
            # Boost_PYTHON*_FOUND / *_LIBRARY_RELEASE variables, so if the
            # chains above found nothing, detect the imported boost::python
            # target directly (the same way ct's python detection does).
            if(NOT Boost_PYTHON_FOUND)
                find_package(Boost QUIET COMPONENTS python3)
                if(TARGET Boost::python3)
                    set(Boost_PYTHON_FOUND TRUE)
                    set(Boost_PYTHON_LIBRARY_RELEASE "Boost::python3" CACHE PATH "" FORCE)
                endif()
            endif()

            if(Boost_PYTHON_FOUND)
                find_package(NumPy 1.7.1)
                if(NUMPY_FOUND)
                    set(MO_HAVE_PYTHON 1 CACHE BOOL INTERNAL FORCE)
                    set(MO_PYTHON_STATUS "${PYTHONLIBS_VERSION_STRING} \n    include:     ${PYTHON_INCLUDE_DIR} \n    Boost:       ${Boost_PYTHON_LIBRARY_RELEASE} \n    Numpy:       ${NUMPY_VERSION}")
                else()
                    set(MO_PYTHON_STATUS "Unable to find numpy")
                endif()
            else()
                set(MO_PYTHON_STATUS "Unable to find a suitable version of boost python ${searched_libs}")
            endif()
        endif(WIN32)
    else(PythonLibs_FOUND AND PythonInterp_FOUND)
        set(MO_PYTHON_STATUS "Unable to find PythonLibs and PythonInterp")
    endif()

else(WITH_PYTHON)
    set(MO_PYTHON_STATUS "Python disabled by WITH_PYTHON cmake flag")
endif()
