include(cmake/dependencies/boost.cmake)
include(cmake/dependencies/cereal.cmake)
include(cmake/dependencies/opencv.cmake)
include(cmake/dependencies/cuda.cmake)
include(cmake/dependencies/rcc.cmake)
include(cmake/dependencies/spdlog.cmake)
set(CT_HAVE_PYTHON 0 CACHE BOOL INTERNAL FORCE)
add_subdirectory(dependencies/ct)
include(dependencies/ct/cmake/python.cmake)




