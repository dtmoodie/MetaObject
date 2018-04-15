file(GLOB_RECURSE src "src/*.cpp" "src/*.hpp" "src/*.h" "src/*.cu")
metaobject_declare_module(NAME params SRC src DEPENDS metaobject_core ct metaobject_types)
target_include_directories(metaobject_params PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>)
target_include_directories(metaobject_params PUBLIC $<BUILD_INTERFACE:${cereal_INCLUDE_DIRS}>)
set(metaobject_param_include
"${metaobject_param_include}
#include <MetaObject/params/buffers/CircularBuffer.hpp>
#include <MetaObject/params/buffers/StreamBuffer.hpp>
#include <MetaObject/params/buffers/Map.hpp>
#include <MetaObject/serialization/CerealPolicy.hpp>
#include <MetaObject/serialization/cereal_map.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/binary.hpp>

" CACHE INTERNAL "" FORCE)

IF(Qt5_FOUND)
  TARGET_LINK_LIBRARIES(metaobject_params Qt5::Core)
  TARGET_INCLUDE_DIRECTORIES(metaobject_params PRIVATE $<BUILD_INTERFACE:${Qt5Core_INCLUDE_DIRS}>)
  set(metaobject_param_include
  "${metaobject_param_include}
#ifdef HAVE_QT
    #include <MetaObject/params/ui/Qt/OpenCV.hpp>
    #include <MetaObject/params/ui/Qt/Containers.hpp>
    #include <MetaObject/params/ui/Qt/TParamProxy.hpp>
#endif
" CACHE INTERNAL "" FORCE)
ENDIF()

add_executable(mo_test_polymorphic_input tests/polymorphic_input.cpp)
target_link_libraries(mo_test_polymorphic_input metaobject_params)
set_property(TARGET mo_test_polymorphic_input PROPERTY CXX_STANDARD 11)

