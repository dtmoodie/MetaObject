#pragma once

#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__) && defined metaobject_metaparams_EXPORTS
  #define METAPARAMTERS_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
  #define METAPARAMTERS_EXPORTS __attribute__ ((visibility ("default")))
#else
  #define METAPARAMTERS_EXPORTS
#endif

#ifdef _MSC_VER
  #ifndef metaobject_metaparams_EXPORTS
    #ifdef _DEBUG
      #pragma comment(lib, "metaobject_metaparamsd.lib")
    #else
      #pragma comment(lib, "metaobject_metaparams.lib")
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
