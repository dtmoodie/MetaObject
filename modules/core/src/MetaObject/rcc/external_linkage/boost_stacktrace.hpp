#ifndef MO_RCC_EXTERN_BOOST_STACKTRACE_HPP
#define MO_RCC_EXTERN_BOOST_STACKTRACE_HPP

#include "RuntimeObjectSystem/RuntimeLinkLibrary.h"
#include <boost/stacktrace.hpp>

#ifndef _WIN32
RUNTIME_COMPILER_LINKLIBRARY("-ldl")
#endif

#endif // MO_RCC_EXTERN_BOOST_STACKTRACE_HPP
