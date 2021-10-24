#ifndef MO_RCC_EXTERN_BOOST_THREAD_HPP
#define MO_RCC_EXTERN_BOOST_THREAD_HPP

#include <boost/thread.hpp>

#define BOOST_MODULE "thread"
#include "boost_base.hpp"
#undef BOOST_MODULE
RUNTIME_COMPILER_LINKLIBRARY("-lpthread")

#endif // MO_RCC_EXTERN_BOOST_THREAD_HPP
