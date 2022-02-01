find_package(cereal QUIET)

if(cereal_FOUND)
    add_library(cereal IMPORTED INTERFACE)
    target_include_directories(cereal
        INTERFACE
            ${cereal_INCLUDE_DIRS}
    )
endif()

if(NOT cereal_FOUND)
    set(JUST_INSTALL_CEREAL ON CACHE BOOL "" FORCE)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/dependencies/cereal")
endif()

