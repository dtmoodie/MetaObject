#pragma once

#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__) && defined MetaParameters_EXPORTS
  #define METAPARAMTERS_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
  #define METAPARAMTERS_EXPORTS __attribute__ ((visibility ("default")))
#else
  #define METAPARAMTERS_EXPORTS
#endif

#ifdef _MSC_VER
  #ifndef MetaParameters_EXPORTS
    #ifdef _DEBUG
      #pragma comment(lib, "MetaParametersd.lib")
    #else
      #pragma comment(lib, "MetaParameters.lib")
    #endif
  #endif
#endif
namespace mo
{
    namespace MetaParams
    {
        METAPARAMTERS_EXPORTS void initialize();
    }
}
