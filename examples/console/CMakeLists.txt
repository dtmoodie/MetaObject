file(GLOB src "*.cpp")
file(GLOB hdr "*.hpp")

add_executable(console ${hdr} ${src})
target_link_libraries(console MetaObject)
set_target_properties(console PROPERTIES FOLDER examples)
if(MSVC)
      CONFIGURE_FILE("../../tests/Test.vcxproj.user.in" ${CMAKE_BINARY_DIR}/examples/console/console.vcxproj.user @ONLY)
endif()