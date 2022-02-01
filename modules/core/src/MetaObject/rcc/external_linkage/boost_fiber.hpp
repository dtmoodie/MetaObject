#ifndef MO_RCC_EXTERN_BOOST_FIBER_HPP
#define MO_RCC_EXTERN_BOOST_FIBER_HPP

#include <boost/fiber/fiber.hpp>

#define BOOST_MODULE "fiber"
#include "boost_base.hpp"
#undef BOOST_MODULE

#define BOOST_MODULE "context"
#include "boost_base.hpp"
#undef BOOST_MODULE

#endif // MO_RCC_EXTERN_BOOST_FIBER_HPP
