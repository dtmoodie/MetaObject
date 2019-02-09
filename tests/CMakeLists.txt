# ----------------------- tests -----------------------------------
find_package(OpenCV QUIET COMPONENTS cudaarithm)
if(Boost_UNIT_TEST_FRAMEWORK_FOUND AND BUILD_TESTS)
    include(CTest)
    enable_testing()
    SUBDIRLIST(tests "${CMAKE_CURRENT_LIST_DIR}")
    foreach(test ${tests})
        if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/${test}/CMakeLists.txt")
            add_subdirectory(${test})
        else()
            file(GLOB_RECURSE test_srcs "${test}/*.cpp")
            if(test_srcs)
                IF(CUDA_FOUND)
                    file(GLOB_RECURSE test_knl "${test}/*.cu")
                ENDIF()
                LIST(LENGTH test_knl num_knl)
                IF(${num_knl} GREATER 0)
                    set(CUDA_NVCC_FLAGS "-std=c++11")
                    cuda_add_executable(test_mo_${test} ${test_srcs} ${test_knl})
                else()
                    add_executable(test_mo_${test} ${test_srcs})
                endif()

                TARGET_LINK_LIBRARIES(test_mo_${test}
                    ${MetaObject_LIBRARIES}
                    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
                    metaobject_params
                    metaobject_core
                )

                if(OpenCV_FOUND)
                    TARGET_LINK_LIBRARIES(test_mo_${test} ${OpenCV_LIBS})
                endif()
                ADD_DEPENDENCIES(test_mo_${test} ${MetaObject_LIBRARIES})
                set_target_properties(test_mo_${test} PROPERTIES FOLDER Tests/MetaObject)
                add_test(NAME test_mo_${test} COMMAND $<TARGET_FILE:test_mo_${test}>)
                RCC_TARGET_CONFIG(test_mo_${test} plugin_libraries_debug plugin_libraries_release)

                if(MSVC)
                    CONFIGURE_FILE("Test.vcxproj.user.in" ${CMAKE_BINARY_DIR}/test_mo_${test}.vcxproj.user @ONLY)
                    CONFIGURE_FILE("Test.vcxproj.user.in" ${CMAKE_CURRENT_BINARY_DIR}/test_mo_${test}.vcxproj.user @ONLY)
                endif()
            endif()
        endif()
    endforeach()
endif()

