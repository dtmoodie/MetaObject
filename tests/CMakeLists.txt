# ----------------------- tests -----------------------------------
if(Boost_UNIT_TEST_FRAMEWORK_FOUND AND BUILD_TESTS)
  include(CTest)
  enable_testing()
  SUBDIRLIST(tests "${CMAKE_CURRENT_LIST_DIR}")
  foreach(test ${tests})
    file(GLOB_RECURSE test_srcs "${test}/*.cpp")
    IF(CUDA_FOUND)
      file(GLOB_RECURSE test_knl "${test}/*.cu")
    ENDIF()
    LIST(LENGTH test_knl num_knl)
    IF(${num_knl} GREATER 0)
      cuda_add_executable(${test} ${test_srcs} ${test_knl})
    else()
      add_executable(${test} ${test_srcs})
    endif()
	
    TARGET_LINK_LIBRARIES(${test}
		${MetaObject_LIBRARIES}
        MetaParameters
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )

    if(OpenCV_FOUND)
      TARGET_LINK_LIBRARIES(${test} ${OpenCV_LIBS})
    endif()
	
    ADD_DEPENDENCIES(${test} ${MetaObject_LIBRARIES})
    set_target_properties(${test} PROPERTIES FOLDER Tests/MetaObject)
    add_test(NAME ${test} COMMAND $<TARGET_FILE:${test}>)
	
    if(MSVC)
      CONFIGURE_FILE("Test.vcxproj.user.in" ${CMAKE_BINARY_DIR}/${test}.vcxproj.user @ONLY)
      CONFIGURE_FILE("Test.vcxproj.user.in" ${CMAKE_CURRENT_BINARY_DIR}/${test}.vcxproj.user @ONLY)
    endif()
  endforeach()
endif()