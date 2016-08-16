
add_library(MetaPython SHARED "MetaPython.cpp")
target_link_libraries(MetaPython 
    MetaObject 
    ${Boost_PYTHON3_LIBRARY_DEBUG} 
    ${Boost_PYTHON3_LIBRARY_RELEASE}
    ${PYTHON_DEBUG_LIBRARY}
    ${PYTHON_LIBRARY})