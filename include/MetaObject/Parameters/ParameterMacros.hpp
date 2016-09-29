#pragma once
#ifndef __CUDACC__
#include "MetaObject/Parameters/TypedParameterPtr.hpp"
#include "MetaObject/Parameters/TypedInputParameter.hpp"
#include "detail/ParameterMacrosImpl.hpp"


#define STRINGIFY_1(X1) #X1
#define STRINGIFY_2(X1, X2) #X1, #X2
#define STRINGIFY_3(X1, X2, X3) #X1, #X2, #X3
#define STRINGIFY_4(X1, X2, X3, X4) #X1, #X2, #X3, #X4
#define STRINGIFY_5(X1, X2, X3, X4, X5) #X1, #X2, #X3, #X4, #X5
#define STRINGIFY_6(X1, X2, X3, X4, X5, X6) #X1, #X2, #X3, #X4, #X5, #X6
#define STRINGIFY_7(X1, X2, X3, X4, X5, X6, X7) #X1, #X2, #X3, #X4, #X5, #X6, #X7

#ifdef _MSC_VER
#define STRINGIFY(...)  BOOST_PP_CAT(BOOST_PP_OVERLOAD(STRINGIFY_, __VA_ARGS__)(__VA_ARGS__), BOOST_PP_EMPTY())
#else
#define STRINGIFY(...)  BOOST_PP_OVERLOAD(STRINGIFY_, __VA_ARGS__)(__VA_ARGS__), BOOST_PP_EMPTY()
#endif

#define ENUM(...) {STRINGIFY(__VA_ARGS__)}, {__VA_ARGS__}


#define PARAM(type, name, init) \
mo::TypedParameterPtr<type> name##_param; \
type name = init; \
PARAM_(type, name, init, __COUNTER__)


#define ENUM_PARAM(name, ...) \
mo::TypedParameterPtr<mo::EnumParameter> name##_param; \
mo::EnumParameter name; \
ENUM_PARAM_(__COUNTER__, name, __VA_ARGS__)


#define RANGED_PARAM(type, name, init, min, max)

#define INPUT(type, name, init) \
type* name = init; \
mo::TypedInputParameterPtr<type> name##_param; \
void init_parameters_(bool firstInit, mo::_counter_<__COUNTER__> dummy) \
{ \
    name##_param.SetMtx(_mtx); \
    name##_param.SetUserDataPtr(&name); \
    name##_param.SetName(#name); \
    AddParameter(&name##_param); \
    init_parameters_(firstInit, --dummy); \
}


#define PROPERTY(type, name, init) \
type name = init; \
SERIALIZE_(name, __COUNTER__)

#define STATUS(type, name, init)\
mo::TypedParameterPtr<type> name##_param; \
type name = init; \
STATUS_(type, name, init, __COUNTER__)

#define TOOLTIP(name, TOOLTIP) TOOLTIP_(name, TOOLTIP, __COUNTER__)

#define DESCRIPTION(name, DESCRIPTION)

#define OUTPUT(type, name, init) \
mo::TypedParameterPtr<type> name##_param; \
OUTPUT_(type, name, init, __COUNTER__); \
type name = init;

#else
#define PARAM(type, name, init)
#define PROPERTY(type, name, init)
#define INPUT(type, name, init)
#define OUTPUT(type, name, init)
#endif
