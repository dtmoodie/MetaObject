FILE(GLOB_RECURSE src "src/*.hpp" "src/*.cu" "src/*.cpp")
metaobject_declare_module(NAME types SRC src DEPENDS ct)