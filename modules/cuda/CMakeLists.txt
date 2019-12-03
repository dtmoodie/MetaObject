FILE(GLOB_RECURSE src "src/*.hpp" "src/*.cu" "src/*.cpp")

find_package(CUDA QUIET)

if(CUDA_FOUND)
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        set(CUDA_PROPAGATE_HOST_FLAGS ON)
        set(CMAKE_CXX_FLAGS "-std=c++11")
    endif()


    metaobject_declare_module(NAME cuda SRC src DEPENDS
        metaobject_core
        metaobject_types
    )

    target_link_libraries(metaobject_cuda
        ${CUDA_CUDART_LIBRARY}
    )

    if(BUILD_TESTS)
        if(WITH_OPENCV)
            find_package(OpenCV QUIET COMPONENTS cudaarithm)
        endif()

        file(GLOB_RECURSE tests "tests/*.cpp" "tests/*.cu")
        cuda_add_executable(test_mo_cuda ${tests})
        target_include_directories(test_mo_cuda
            PUBLIC SYSTEM ${Boost_INCLUDE_DIR}    
        )
        if(OpenCV_FOUND)
            target_compile_definitions(test_mo_cuda
                PRIVATE MO_HAVE_OPENCV
            )
            target_include_directories(test_mo_cuda
                PRIVATE ${OpenCV_INCLUDE_DIRS}
            )
        endif()

        target_link_libraries(test_mo_cuda metaobject_cuda ${OpenCV_LIBRARIES} gtest gtest_main)
        add_test(NAME test_mo_cuda COMMAND test_mo_cuda)

        cuda_add_executable(test_mo_cuda_fiber tests2/cuda_build.cu)
        TARGET_INCLUDE_DIRECTORIES(test_mo_cuda_fiber PRIVATE "../core/src")
        TARGET_INCLUDE_DIRECTORIES(test_mo_cuda_fiber PRIVATE "../core/jetson_patch")
        target_link_libraries(test_mo_cuda_fiber gtest gtest_main )
        if(TARGET Boost::system)
            target_link_libraries(test_mo_cuda_fiber Boost::system)
        endif()
        if(Boost_INCLUDE_DIR)
            target_include_directories(test_mo_cuda_fiber
                PUBLIC SYSTEM ${Boost_INCLUDE_DIR}
            )
        endif()
        
        set_target_properties(test_mo_cuda PROPERTIES FOLDER Tests/MetaObject/cuda)
        set_target_properties(test_mo_cuda_fiber PROPERTIES FOLDER Tests/MetaObject/cuda_fiber)
        setup_msvc_target(test_mo_cuda)
        setup_msvc_target(test_mo_cuda_fiber)

    endif()
endif()
