#include "RuntimeObjectSystem/RuntimeLinkLibrary.h"

#ifdef _WIN32
#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_core" CV_VERSION_ "d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_core" CV_VERSION_ ".lib")
#endif
#else
RUNTIME_COMPILER_LINKLIBRARY("-lboost_" BOOST_MODULE)
#endif
