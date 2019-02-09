function(metaobject_declare_module)
    set(oneValueArgs NAME)
    set(multiValueArgs SRC DEPENDS FLAGS CUDA_SRC)
    cmake_parse_arguments(metaobject_declare_module "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    GroupSources("${CMAKE_CURRENT_LIST_DIR}/src/MetaObject/" "/" " ")
    if(${metaobject_declare_module_CUDA_SRC})
        cuda_add_library(metaobject_${metaobject_declare_module_NAME} SHARED ${${metaobject_declare_module_CUDA_SRC}})
    else()
        if(${metaobject_declare_module_SRC})
            add_library(metaobject_${metaobject_declare_module_NAME} SHARED ${${metaobject_declare_module_SRC}})
        else()
            file(GLOB_RECURSE src "src/*.cpp" "src/*.h" "src/*.hpp")
            add_library(metaobject_${metaobject_declare_module_NAME} SHARED ${src})
        endif()
    endif()
    
    set_target_properties(metaobject_${metaobject_declare_module_NAME} PROPERTIES LINKER_LANGUAGE CXX)
    
    set(metaobject_modules "${metaobject_modules};metaobject_${metaobject_declare_module_NAME}" CACHE INTERNAL "" FORCE)

    target_compile_definitions(metaobject_${metaobject_declare_module_NAME} PRIVATE -DMetaObject_EXPORTS)

    if(metaobject_declare_module_DEPENDS)
        rcc_link_lib(metaobject_${metaobject_declare_module_NAME} ${metaobject_declare_module_DEPENDS})
    endif()



    if(UNIX)
        target_compile_options(metaobject_${metaobject_declare_module_NAME} PUBLIC "-fPIC;-Wl,--no-undefined")
    endif()
    set(metaobject_${metaobject_declare_module_NAME}_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/src" CACHE INTERNAL "" FORCE)
    target_include_directories(metaobject_${metaobject_declare_module_NAME}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
            $<INSTALL_INTERFACE:include>
    )
    set(metaobject_module_includes "${metaobject_module_includes};${CMAKE_CURRENT_LIST_DIR}/src" CACHE INTERNAL "" FORCE)

    set_target_properties(metaobject_${metaobject_declare_module_NAME} PROPERTIES FOLDER Modules)
    if(metaobject_declare_module_FLAGS)
        target_compile_options(metaobject_${metaobject_declare_module_NAME} PUBLIC ${metaobject_declare_module_FLAGS})
    endif()

    export(TARGETS metaobject_${metaobject_declare_module_NAME}
        FILE "${PROJECT_BINARY_DIR}/MetaObjectTargets-${metaobject_declare_module_NAME}.cmake"
    )
    install(TARGETS metaobject_${metaobject_declare_module_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
    install(TARGETS metaobject_${metaobject_declare_module_NAME}
        DESTINATION lib
        EXPORT metaobject_${metaobject_declare_module_NAME}Targets
    )
    install(DIRECTORY src/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
    )

    install(DIRECTORY src/
        DESTINATION include
        FILES_MATCHING PATTERN "*.hpp"
    )
    install(EXPORT metaobject_${metaobject_declare_module_NAME}Targets DESTINATION "${CMAKE_INSTALL_PREFIX}/share/MetaObject" COMPONENT dev)

    if(RCC_VERBOSE_CONFIG)
        set(test_lib "")
        set(test_inc "")
        set(test_lib "")
        _target_helper(test_lib test_inc test_lib metaobject_${metaobject_declare_module_NAME} " ")
        message("---------------")
        list(REMOVE_DUPLICATES test_lib)
        foreach(lib ${test_lib})
            message("  ${lib}")
        endforeach()
        list(REMOVE_DUPLICATES test_inc)
        foreach(inc ${test_inc})
            message("  ${inc}")
        endforeach()
    endif(RCC_VERBOSE_CONFIG)
endfunction()
