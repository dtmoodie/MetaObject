if(WITH_QT)
  find_package(Qt5 QUIET COMPONENTS Core Widgets Gui)
  if(Qt5_FOUND)
    set(AUTOMOC OFF)
    set(MO_HAVE_QT 1 CACHE BOOL INTERNAL FORCE)
    list(APPEND src ${MOC})
    list(APPEND link_libs Qt5::Core Qt5::Widgets Qt5::Gui)
    get_target_property(qt5_core_bin_ Qt5::Core IMPORTED_LOCATION_DEBUG)
    get_filename_component(qt5_bin_dir_ "${qt5_core_bin_}" DIRECTORY)
    list(APPEND PROJECT_BIN_DIRS_DEBUG ${qt5_bin_dir_})
    list(APPEND PROJECT_BIN_DIRS_RELEASE ${qt5_bin_dir_})
    list(APPEND PROJECT_BIN_DIRS_RELWITHDEBINFO ${qt5_bin_dir_})
    set(Qt5_BIN_DIR_OPT ${qt5_bin_dir_} CACHE PATH "" FORCE)
    set(Qt5_BIN_DIR ${qt5_bin_dir_} CACHE PATH "" FORCE)
    set(Qt5_BIN_DIR_DBG ${qt5_bin_dir_} CACHE PATH "" FORCE)
    set(bin_dirs_ "${BIN_DIRS};Qt5")
    list(REMOVE_DUPLICATES bin_dirs_)
    set(BIN_DIRS "${bin_dirs_}" CACHE STRING "" FORCE)
    rcc_find_path(Qt5_PLUGIN_PATH qwindows.dll PATHS ${Qt5_BIN_DIR}/../plugins/platforms)
  endif()
endif()
