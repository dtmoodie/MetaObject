if(NOT BUILD_DEPENDENCIES)
    find_package(spdlog QUIET)
endif()
if(RCC_FOUND)
    message(STATUS "spdlog found at: ${spdlog_DIR}")
else(RCC_FOUND)
    ADD_SUBDIRECTORY("dependencies/spdlog")
endif(RCC_FOUND)
