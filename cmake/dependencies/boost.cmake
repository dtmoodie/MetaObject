if(Boost_DIR)
    set(BOOST_ROOT ${Boost_DIR})
    if(EXISTS "${Boost_DIR}/stage/lib")
        set(BOOST_LIBRARYDIR "${Boost_DIR}/stage/lib")
    endif()
endif()

# No 'system': Boost.System has been header-only (empty library) since
# Boost 1.69, and Ubuntu 26.04's Boost 1.90 packaging no longer ships the
# stub libboost_system at all (no package provides it), so requiring it
# breaks configure there. Nothing links against real code in it.
set(Boost_required_components thread fiber filesystem chrono context)
set(Boost_USE_STATIC_LIBS        OFF)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME     OFF)
ADD_DEFINITIONS(-DBOOST_ALL_DYN_LINK)
# Force the classic FindBoost module search instead of the boost-cmake
# package-configuration (CONFIG) search. Newer distro packaging (Ubuntu
# 26.04's Boost 1.90) ships a BoostConfig.cmake whose per-component
# resolution (find_package(boost_system), ...) finds no matching
# boost_<cmp>Config.cmake anywhere in the distribution, so CONFIG mode
# dies at configure; the module path (headers + version-suffixed libs)
# works on all supported distros.
set(Boost_NO_BOOST_CMAKE ON)
find_package(Boost REQUIRED COMPONENTS ${Boost_required_components})
link_directories(${Boost_LIBRARY_DIR_DEBUG})
link_directories(${Boost_LIBRARY_DIR})
list(APPEND PROJECT_BIN_DIRS_DEBUG_ ${Boost_LIBRARY_DIR})
list(APPEND PROJECT_BIN_DIRS_RELEASE_ ${Boost_LIBRARY_DIR})
list(APPEND PROJECT_BIN_DIRS_RELWITHDEBINFO_ ${Boost_LIBRARY_DIR})
list(APPEND PROJECT_BIN_DIRS_DEBUG_ ${Boost_LIBRARY_DIR_DEBUG})
list(APPEND PROJECT_BIN_DIRS_RELEASE_ ${Boost_LIBRARY_DIR_RELEASE})
list(APPEND PROJECT_BIN_DIRS_RELWITHDEBINFO_ ${Boost_LIBRARY_DIR_RELEASE})
set(bin_dirs_ "${BIN_DIRS};Boost")
list(REMOVE_DUPLICATES bin_dirs_)
set(BIN_DIRS "${bin_dirs_}" CACHE STRING "" FORCE)
set(Boost_TARGETS "")
foreach(cmp ${Boost_required_components})
    if(NOT TARGET Boost::${cmp})
        STRING(TOUPPER ${cmp} CMP)
        if(Boost_${CMP}_LIBRARY_DEBUG OR Boost_${CMP}_LIBRARY_RELEASE)
            add_library(Boost::${cmp} IMPORTED SHARED)
            set_target_properties(Boost::${cmp} 
                PROPERTIES
                    IMPORTED_LOCATION_RELEASE ${Boost_${CMP}_LIBRARY_RELEASE}
                    IMPORTED_LOCATION_DEBUG ${Boost_${CMP}_LIBRARY_DEBUG}
                    INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIR}
            )
        else()
            message("Boost target Boost::${cmp} not defined")
        endif()
    endif()
    if(TARGET Boost::${cmp})
        list(APPEND Boost_TARGETS "Boost::${cmp}")
    endif()
endforeach()
