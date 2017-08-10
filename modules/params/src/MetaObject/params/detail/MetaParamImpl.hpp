#pragma once
#include "MetaObject/params/TInputParam.hpp"
#include "MetaObject/params/TParamPtr.hpp"
#include "MetaObject/params/TParam.hpp"
#include "MetaObject/params/detail/TInputParamPtrImpl.hpp"
#include "MetaObject/params/detail/TParamPtrImpl.hpp"
#include <MetaObject/params/detail/ParamMacrosImpl.hpp>
#include "MetaObject/params/detail/TParamImpl.hpp"

#define INSTANTIATE_META_PARAM(TYPE)                                                                               \
    static mo::MetaParam<mo::argument_type<void(TYPE)>::type, __COUNTER__> COMBINE(g_meta_Param, __LINE__)(#TYPE); \
    template class mo::TInputParamPtr<mo::argument_type<void(TYPE)>::type>;                                        \
    template class mo::TParamPtr<mo::argument_type<void(TYPE)>::type>;                                             \
    template class mo::TParam<mo::argument_type<void(TYPE)>::type>;                                                \
    template class mo::TParamOutput<mo::argument_type<void(TYPE)>::type>
