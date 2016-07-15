#pragma once
#include <boost/preprocessor.hpp>
#include "detail/ParameterMacrosImpl.hpp"
#include "MetaObject/Parameters/TypedParameterPtr.hpp"

#define PARAM_(...) BOOST_PP_CAT( BOOST_PP_OVERLOAD(PARAM___, __VA_ARGS__ )(__VA_ARGS__), BOOST_PP_EMPTY() )



#define PARAM(type, name, init) \
std::shared_ptr<mo::TypedParameterPtr<type>> name##_param; \
type name = init; \
PARAM__(type, name, init, __COUNTER__)

#define RANGED_PARAM(type, name, init, min, max)
