FILE(GLOB_RECURSE src "src/*.hpp" "src/*.cu" "src/*.cpp")
metaobject_declare_module(NAME core CUDA_SRC src DEPENDS ${OpenCV_LIBS} ${Boost_LIBRARIES})
