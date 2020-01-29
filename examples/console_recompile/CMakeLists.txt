file(GLOB src "*.cpp")
file(GLOB hdr "*.hpp")

add_executable(console_recompile ${hdr} ${src})
target_link_libraries(console_recompile ${MetaObject_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenCV_LIBS})
set_target_properties(console_recompile PROPERTIES FOLDER examples)
if(MSVC)
      setup_msvc_target(console_recompile)
endif()
