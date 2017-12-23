if(WITH_ZEROMQ)
    find_package(ZeroMQ QUIET)
    if(${ZeroMQ_FOUND})
      include_directories(${ZeroMQ_INCLUDE_DIR})
      include_directories("${CMAKE_CURRENT_LIST_DIR}/dependencies/cppzmq") # cpp bindings
      list(APPEND link_libs "optimized;${ZeroMQ_LIBRARY_RELEASE};debug;${ZeroMQ_LIBRARY_DEBUG}")
      add_definitions(-DHAVE_ZEROMQ)
    endif(${ZeroMQ_FOUND})
endif(WITH_ZEROMQ)
