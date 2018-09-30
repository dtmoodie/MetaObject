file(GLOB_RECURSE src "src/*.cpp" "src/*.hpp" "src/*.h" "src/*.cu")
metaobject_declare_module(NAME object SRC src DEPENDS metaobject_core metaobject_params RuntimeObjectSystem RuntimeCompiler)