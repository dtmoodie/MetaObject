#pragma once

#ifdef MO_EXPORTS
#undef MO_EXPORTS
#endif
#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__) && defined MetaObject_EXPORTS
#  define MO_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#  define MO_EXPORTS __attribute__ ((visibility ("default")))
#else
#  define MO_EXPORTS
#endif

#ifndef MetaObject_EXPORTS
#include "RuntimeLinkLibrary.h"
  #ifdef WIN32
    #pragma comment(lib, "Advapi32.lib")
    #ifdef _DEBUG
      RUNTIME_COMPILER_LINKLIBRARY("MetaObjectd.lib")
      RUNTIME_COMPILER_LINKLIBRARY("RuntimeCompilerd.lib")
      RUNTIME_COMPILER_LINKLIBRARY("RuntimeObjectSystemd.lib")
    #else
      RUNTIME_COMPILER_LINKLIBRARY("MetaObject.lib")
      RUNTIME_COMPILER_LINKLIBRARY("RuntimeCompiler.lib")
      RUNTIME_COMPILER_LINKLIBRARY("RuntimeObjectSystem.lib")
    #endif
  #else
    #ifdef _DEBUG
      RUNTIME_COMPILER_LINKLIBRARY("-lMetaObjectd")
      RUNTIME_COMPILER_LINKLIBRARY("-lRuntimeCompilerd")
      RUNTIME_COMPILER_LINKLIBRARY("-lRuntimeObjectSystemd")
    #else
      RUNTIME_COMPILER_LINKLIBRARY("-lMetaObject")
      RUNTIME_COMPILER_LINKLIBRARY("-lRuntimeCompiler")
      RUNTIME_COMPILER_LINKLIBRARY("-lRuntimeObjectSystem")
    #endif
  #endif
#endif