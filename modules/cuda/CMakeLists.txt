FILE(GLOB_RECURSE src "src/*.hpp" "src/*.cu" "src/*.cpp")

find_package(CUDA QUIET)

if(CUDA_FOUND)
    # nvcc's host compiler defaults to CMAKE_CXX_COMPILER, so the host side
    # of .cu files inherits this directory flag via host-flag propagation.
    # CUDA 13's CCCL headers (thrust/cu++/cub) #error out unless the host
    # dialect is at least C++17, so bump the flag there (CUDA 11/12 are happy
    # with c++11). Clang only joins the CUDA >= 13 branch: clang + CUDA 11/12
    # builds green without this gate (nvcc's default host dialect suffices
    # for those CCCL versions), so that path is left untouched.
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU"
        OR (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang" AND CUDA_VERSION_MAJOR GREATER_EQUAL 13))
        set(CUDA_PROPAGATE_HOST_FLAGS ON)
        if(CUDA_VERSION_MAJOR GREATER_EQUAL 13)
            set(CMAKE_CXX_FLAGS "-std=c++17")
        else()
            set(CMAKE_CXX_FLAGS "-std=c++11")
        endif()
    endif()

    # CUDA 13's CCCL headers require C++17 in BOTH compilation phases. The
    # host phase gets it from the directory flag above (host-flag
    # propagation), but the device phase (nvcc's own frontend, which parses
    # the whole TU including host headers) uses nvcc's INTERNAL notion of
    # the host compiler's default dialect: c++17 for recent g++, but only
    # c++11 for clang. A c++11 device parse rejects the in-class C++17 CTAD
    # deduction guides that libstdc++ >= 12 headers carry (the ones clang
    # picks up; gcc -ccbin parses the older out-of-class form). Force the
    # device dialect explicitly.
    if(CUDA_VERSION_MAJOR GREATER_EQUAL 13)
        list(APPEND CUDA_NVCC_FLAGS --std=c++17)
    endif()


    metaobject_declare_module(NAME cuda SRC src DEPENDS
        metaobject_core
        metaobject_types
    )

    target_link_libraries(metaobject_cuda
        PUBLIC
            ${CUDA_CUDART_LIBRARY}
    )
    target_include_directories(metaobject_cuda PUBLIC ${CUDA_INCLUDE_DIRS})
    # CUDA 13 moved thrust/cub/libcu++ under <include>/cccl and dropped the
    # top-level thrust/ shim, so plain C++ translation units (which do not
    # get nvcc's implicit cccl search path) no longer find <thrust/...>.
    # Guarded by EXISTS: a no-op on CUDA 11/12, where the headers sit at
    # the top of the include dir.
    if(CUDA_TOOLKIT_INCLUDE AND EXISTS "${CUDA_TOOLKIT_INCLUDE}/cccl")
        target_include_directories(metaobject_cuda PUBLIC "${CUDA_TOOLKIT_INCLUDE}/cccl")
    endif()

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

        target_link_libraries(test_mo_cuda PRIVATE metaobject_cuda ${OpenCV_LIBRARIES} gtest gtest_main)
        add_test(NAME test_mo_cuda COMMAND test_mo_cuda)

        cuda_add_executable(test_mo_cuda_fiber tests2/cuda_build.cu)
        TARGET_INCLUDE_DIRECTORIES(test_mo_cuda_fiber PRIVATE "../core/src")
        target_link_libraries(test_mo_cuda_fiber PRIVATE gtest gtest_main )
        if(TARGET Boost::system)
            target_link_libraries(test_mo_cuda_fiber PRIVATE Boost::system)
        endif()
        if(Boost_INCLUDE_DIR)
            target_include_directories(test_mo_cuda_fiber
                PUBLIC SYSTEM ${Boost_INCLUDE_DIR}
            )
        endif()
        
        set_target_properties(test_mo_cuda PROPERTIES FOLDER Tests/MetaObject/cuda)
        set_target_properties(test_mo_cuda_fiber PROPERTIES FOLDER Tests/MetaObject/cuda_fiber)
        # The vendored googletest requires C++17, but these targets inherit the
        # directory-level -std=c++11 (needed for nvcc host-flag propagation).
        # CMake >= 3.20 suppresses the CXX_STANDARD flag entirely when
        # CMAKE_CXX_FLAGS already carries a -std=, so set_target_properties
        # alone is not enough (worked on CMake 3.16/Ubuntu 20.04, broke on
        # CMake 3.22/Ubuntu 22.04). Add the standard as a compile option too:
        # user compile options are appended after the directory flags, so
        # -std=c++17 wins for the C++ sources; the .cu files keep the C++11
        # host standard via host-flag propagation.
        set_target_properties(test_mo_cuda test_mo_cuda_fiber PROPERTIES CXX_STANDARD 17)
        target_compile_options(test_mo_cuda PRIVATE -std=c++17)
        target_compile_options(test_mo_cuda_fiber PRIVATE -std=c++17)
        setup_msvc_target(test_mo_cuda)
        setup_msvc_target(test_mo_cuda_fiber)
        add_test(NAME test_mo_cuda_fiber COMMAND test_mo_cuda_fiber)

    endif()
endif()
