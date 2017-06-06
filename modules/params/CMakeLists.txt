file(GLOB_RECURSE src "src/*.cpp" "src/*.hpp" "src/*.h" "src/*.cu")
metaobject_declare_module(NAME params SRC src DEPENDS metaobject_core)
target_include_directories(metaobject_params PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>)
target_include_directories(metaobject_params PUBLIC $<BUILD_INTERFACE:${cereal_INCLUDE_DIRS}>)
IF(Qt5_FOUND)
  TARGET_LINK_LIBRARIES(metaobject_params Qt5::Core)
  TARGET_INCLUDE_DIRECTORIES(metaobject_params PRIVATE $<BUILD_INTERFACE:${Qt5Core_INCLUDE_DIRS}>)
ENDIF()


