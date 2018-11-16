FILE(GLOB_RECURSE src "src/*.hpp" "src/*.cu" "src/*.cpp")

metaobject_declare_module(NAME core SRC src DEPENDS 
    ${OpenCV_LIBS} 
    Boost::thread
    Boost::fiber
    Boost::system
    cereal 
    RuntimeObjectSystem
)

target_include_directories(metaobject_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
)
if(${MO_HAVE_CUDA})
    target_link_libraries(metaobject_core ${CUDA_CUDART_LIBRARY})
endif(${MO_HAVE_CUDA})

if(BUILD_TESTS)
    file(GLOB_RECURSE tests "tests/*.cpp")
    add_executable(mo_test_core ${tests})
    target_link_libraries(mo_test_core metaobject_core)
    set_property(TARGET mo_test_core PROPERTY CXX_STANDARD 11)
endif()
