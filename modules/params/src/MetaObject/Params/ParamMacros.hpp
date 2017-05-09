#pragma once
#ifndef __CUDACC__
#include "MetaObject/Params/TParamPtr.hpp"
#include "MetaObject/Params/TInputParam.hpp"
#include "MetaObject/Params/Detail/ParamMacrosImpl.hpp"
#include "MetaObject/Detail/HelperMacros.hpp"

#define PARAM(type_, name, init) \
mo::TParamPtr<mo::argument_type<void(type_)>::type> name##_param; \
mo::argument_type<void(type_)>::type name = init; \
PARAM_(type_, name, init, __COUNTER__)


#define ENUM_PARAM(name, ...) \
mo::TParamPtr<mo::EnumParam> name##_param; \
mo::EnumParam name; \
ENUM_PARAM_(__COUNTER__, name, __VA_ARGS__)


#define RANGED_PARAM(type, name, init, min, max)

#define INPUT(type_, name, init) \
const mo::argument_type<void(type_)>::type* name = init; \
mo::TInputParamPtr<mo::argument_type<void(type_)>::type> name##_param; \
void _init_params(bool firstInit, mo::_counter_<__COUNTER__> dummy) \
{ \
    name##_param.setMtx(_mtx); \
    name##_param.setUserDataPtr(&name); \
    name##_param.setName(#name); \
    addParam(&name##_param); \
    _init_params(firstInit, --dummy); \
} \
static void _list_param_info(std::vector<mo::ParamInfo*>& info, mo::_counter_<__COUNTER__> dummy) \
{ \
    static mo::ParamInfo s_info(mo::TypeInfo(typeid(mo::argument_type<void(type_)>::type)), \
                              #name, "", "", mo::Input_e, #init); \
    info.push_back(&s_info); \
    _list_param_info(info, --dummy); \
}

#define OPTIONAL_INPUT(type, name, init) \
INPUT(type, name, init); \
APPEND_FLAGS(name, mo::Optional_e);

#define APPEND_FLAGS(name, flags) \
void _init_params(bool firstInit, mo::_counter_<__COUNTER__> dummy) \
{ \
    _init_params(firstInit, --dummy); \
    name##_param.AppendFlags(flags); \
}


#define PROPERTY(type_, name, init) \
mo::argument_type<void(type_)>::type name; \
void _init_params(bool firstInit, mo::_counter_<__COUNTER__> dummy) \
{ \
    if(firstInit) \
        name = init; \
    _init_params(firstInit, --dummy); \
} \
mo::TParamPtr<mo::argument_type<void(type_)>::type> name##_param; \
SERIALIZE_(name, __COUNTER__)

#define PERSISTENT_(type_, name, N) \
mo::TParamPtr<mo::argument_type<void(type_)>::type> name##_param; \
INIT_(name, N) \
LOAD_SAVE_(name, N)

#define PERSISTENT(type_, name) \
mo::argument_type<void(type_)>::type name; \
PERSISTENT_(type, name, __COUNTER__)

#define INIT(name, init) INIT_(name, init, __COUNTER__)

#define STATUS(type_, name, init)\
mo::TParamPtr<mo::argument_type<void(type_)>::type> name##_param; \
mo::argument_type<void(type_)>::type name = init; \
STATUS_(type, name, init, __COUNTER__)

#define TOOLTIP(name, TOOLTIP) TOOLTIP_(name, TOOLTIP, __COUNTER__)

#define DESCRIPTION(name, DESCRIPTION)

#define OUTPUT(type_, name, init) \
mo::TParamOutput<mo::argument_type<void(type_)>::type> name##_param; \
OUTPUT_(type_, name, init, __COUNTER__); \
mo::argument_type<void(type_)>::type& name = name##_param.reset(init);

#define SOURCE(type, name, init) \
OUTPUT(type, name, init) \
APPEND_FLAGS(name, mo::Source_e)

#else
#define PARAM(type, name, init)
#define PROPERTY(type, name, init)
#define INPUT(type, name, init)
#define OUTPUT(type, name, init)
#endif
