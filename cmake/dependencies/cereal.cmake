find_package(cereal QUIET)

if(NOT TARGET cereal)
    set(JUST_INSTALL_CEREAL ON CACHE BOOL "" FORCE)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/dependencies/cereal")
    set(cereal_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/cereal/include" CACHE PATH "")
endif()
