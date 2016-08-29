#pragma once
#include "MetaObjectMacrosImpl.hpp"


// -------------------------------------------------------------------------------------------
/*#ifdef _MSC_VER
#define MO_BEGIN(...) BOOST_PP_CAT(BOOST_PP_OVERLOAD(MO_BEGIN_, __VA_ARGS__)(__VA_ARGS__, __COUNTER__), BOOST_PP_EMPTY())
#else
#define MO_BEGIN(...) BOOST_PP_OVERLOAD(MO_BEGIN_, __VA_ARGS__)(__VA_ARGS__, __COUNTER__)
#endif*/

#define MO_BEGIN(CLASS_NAME) MO_BEGIN_1(CLASS_NAME, __COUNTER__)
#define MO_DERIVE(CLASS_NAME, BASE_NAME) MO_BEGIN_2(CLASS_NAME, BASE_NAME, __COUNTER__)




#define MO_END MO_END_(__COUNTER__)




