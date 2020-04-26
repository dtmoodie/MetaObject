// clang-format off
#ifndef META_OBJECT_EXPORT
    #define META_OBJECT_EXPORT

    #if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__) && defined(MetaObject_EXPORTS)
        #define MO_EXPORTS __declspec(dllexport)
        #define MO_TEMPLATE_EXTERN
    #elif defined __GNUC__ && __GNUC__ >= 4
        #define MO_EXPORTS __attribute__((visibility("default")))
        #if defined __GNUC__ > 4
        #define MO_TEMPLATE_EXTERN
        #endif
    #else
        #define MO_EXPORTS
    #endif

    #ifndef MO_EXPORTS
        #define MO_EXPORTS
    #endif

    #ifdef _WIN32
        #pragma warning(disable : 4251)
        #pragma warning(disable : 4275)
    #endif

    #if !defined(MetaObject_EXPORTS)
        #include "RuntimeObjectSystem/RuntimeLinkLibrary.h"

        #if defined(_WIN32)
            #pragma comment(lib, "Advapi32.lib")
            #if defined(_DEBUG)
                RUNTIME_COMPILER_LINKLIBRARY("RuntimeCompilerd.lib")
                RUNTIME_COMPILER_LINKLIBRARY("RuntimeObjectSystemd.lib")
            #else
                RUNTIME_COMPILER_LINKLIBRARY("RuntimeCompiler.lib")
                RUNTIME_COMPILER_LINKLIBRARY("RuntimeObjectSystem.lib")
            #endif // _DEBUG
        #else // UNIX
            #if defined(NDEBUG)
                RUNTIME_COMPILER_LINKLIBRARY("-lRuntimeCompiler")
                RUNTIME_COMPILER_LINKLIBRARY("-lRuntimeObjectSystem")
            #else
                RUNTIME_COMPILER_LINKLIBRARY("-lRuntimeCompilerd")
                RUNTIME_COMPILER_LINKLIBRARY("-lRuntimeObjectSystemd")
            #endif
        #endif
    #endif // MetaObject_EXPORTS


    #ifdef __GNUC__
        #if __GNUC__ >= 5
            #define MO_DEPRECATED [[deprecated]]
        #else
            #define MO_DEPRECATED
        #endif
    #else
        #define MO_DEPRECATED [[deprecated]]
    #endif

#endif // META_OBJECT_EXPORT

#ifdef METAOBJECT_MODULE
    #ifndef MetaObject_EXPORTS
        #ifdef _WIN32
            #ifdef _DEBUG
                RUNTIME_COMPILER_LINKLIBRARY("metaobject_" METAOBJECT_MODULE ".lib")
            #else
                RUNTIME_COMPILER_LINKLIBRARY("metaobject_" METAOBJECT_MODULE ".lib")
            #endif
        #else // Unix
            #ifdef NDEBUG
                RUNTIME_COMPILER_LINKLIBRARY("-lmetaobject_" METAOBJECT_MODULE)
            #else
                RUNTIME_COMPILER_LINKLIBRARY("-lmetaobject_" METAOBJECT_MODULE "d")
            #endif
        #endif // WIN32
    #endif // MetaObject_EXPORTS
#endif // METAOBJECT_MODULE

// clang-format on
