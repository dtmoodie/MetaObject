FILE(GLOB_RECURSE src "src/*.hpp" "src/*.cu" "src/*.cpp")
metaobject_declare_module(NAME core CUDA_SRC src DEPENDS ${OpenCV_LIBS})
target_include_directories(metaobject_core PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>)