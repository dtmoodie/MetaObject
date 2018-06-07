#pragma once
#ifndef __CUDACC__
#include "MetaObject/core/detail/HelperMacros.hpp"
#include "MetaObject/params/IParam.hpp"
#include "MetaObject/params/TInputParam.hpp"
#include "MetaObject/params/TParamPtr.hpp"
#include "MetaObject/params/detail/ParamMacrosImpl.hpp"
#include "MetaObject/types/file_types.hpp"

#define PARAM(type_, name, ...)                                                                                        \
    mo::TParamPtr<type_> name##_param;                                                                                 \
    type_ name = __VA_ARGS__;                                                                                          \
    VISIT(name, mo::CONTROL)



#define ENUM_PARAM(name, ...)                                                                                          \
    mo::TParamPtr<mo::EnumParam> name##_param;                                                                         \
    mo::EnumParam name;                                                                                                \
    ENUM_PARAM_(__COUNTER__, name, __VA_ARGS__)

#define RANGED_PARAM(type, name, init, min, max)

#define INPUT(type_, name, init)                                                                                       \
    mo::TInputParamPtr<type_>::Input_t name = init;                                     \
    mo::TInputParamPtr<type_> name##_param; \
    VISIT(name, mo::INPUT)

#define OPTIONAL_INPUT(type, name, init)                                                                               \
    INPUT(type, name, init)                                                                                            \
    APPEND_FLAGS(name, mo::ParamFlags::Optional_e)

#define APPEND_FLAGS(name, flags)                                                                                      \
    void _init_params(bool firstInit, mo::_counter_<__COUNTER__> dummy)                                                \
    {                                                                                                                  \
        _init_params(firstInit, --dummy);                                                                              \
        name##_param.appendFlags(flags);                                                                               \
    }

#define PROPERTY(type_, name, init)                                                                                    \
    mo::argument_type<void(type_)>::type name;                                                                         \
    void _init_params(bool firstInit, mo::_counter_<__COUNTER__> dummy)                                                \
    {                                                                                                                  \
        if (firstInit)                                                                                                 \
            name = init;                                                                                               \
        _init_params(firstInit, --dummy);                                                                              \
    }                                                                                                                  \
    mo::TParamPtr<mo::argument_type<void(type_)>::type> name##_param;                                                  \
    SERIALIZE_(name, __COUNTER__)

#define PERSISTENT_(type_, name, N)                                                                                    \
    mo::TParamPtr<mo::argument_type<void(type_)>::type> name##_param;                                                  \
    INIT_(name, N)                                                                                                     \
    LOAD_SAVE_(name, N)

#define PERSISTENT(type_, name)                                                                                        \
    mo::argument_type<void(type_)>::type name;                                                                         \
    PERSISTENT_(type, name, __COUNTER__)

#define INIT(name, init) INIT_(name, init, __COUNTER__)

#define STATUS(type_, name, init)                                                                                      \
    mo::TParamPtr<mo::argument_type<void(type_)>::type> name##_param;                                                  \
    mo::argument_type<void(type_)>::type name = init;                                                                  \
    STATUS_(type_, name, init, __COUNTER__)

#define TOOLTIP(name, TOOLTIP) TOOLTIP_(name, TOOLTIP, __COUNTER__)

#define DESCRIPTION(name, DESCRIPTION)

#define OUTPUT(type_, name, init)          \
    mo::TParamOutput<type_> name##_param;  \
    type_ name = init;                     \
    VISIT(name, mo::OUTPUT)

#define SOURCE(type, name, init)                                                                                       \
    OUTPUT(type, name, init)                                                                                           \
    APPEND_FLAGS(name, mo::ParamFlags::Source_e)

#else
#define PARAM(type, name, init)
#define PROPERTY(type, name, init)
#define INPUT(type, name, init)
#define OUTPUT(type, name, init)
#endif
