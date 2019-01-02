FILE(GLOB_RECURSE src "src/*.hpp" "src/*.cu" "src/*.cpp")

if(JETSON_COMPATABILITY)
    message(STATUS "Compiling with boost fiber patches for nvidia jetson")
    file(GLOB_RECURSE aux_src "jetson_patch/*.hpp" "jetson_patch/*.cpp")
    LIST(APPEND src ${aux_src})
endif()

metaobject_declare_module(NAME core SRC src DEPENDS
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

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 5.0)
    target_include_directories(metaobject_core
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/jetson_patch>
    )
endif()

if(${MO_HAVE_CUDA})
    target_link_libraries(metaobject_core ${CUDA_CUDART_LIBRARY})
endif(${MO_HAVE_CUDA})

file(GLOB_RECURSE examples "examples/*.cpp")

foreach(example ${examples})
    get_filename_component(name ${example} NAME_WE)

    add_executable(mo_example_${name} ${example})

    target_link_libraries(mo_example_${name} metaobject_core)
endforeach()

if(BUILD_TESTS)
    find_package(OpenCV QUIET)
    file(GLOB_RECURSE tests "tests/*.cpp")
    add_executable(test_mo_core ${tests})
    target_link_libraries(test_mo_core metaobject_core)

    if(OpenCV_FOUND)
        target_compile_definitions(test_mo_core
            PRIVATE HAVE_OPENCV
        )
        target_link_libraries(test_mo_core ${OpenCV_LIBRARIES})
    endif()
    set_property(TARGET test_mo_core PROPERTY CXX_STANDARD 11)
endif()
