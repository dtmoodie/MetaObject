#pragma once
#include "MetaObject/params/TParamPtr.hpp"
#include "MetaObject/params/TSubscriberPtr.hpp"
#include <MetaObject/params/detail/ParamMacrosImpl.hpp>

#define EXTERN_TYPE(TYPE)                                                                                              \
    template class mo::TSubscriberPtr<mo::argument_type<void(TYPE)>::type>;                                            \
    template struct mo::TParamPtr<mo::argument_type<void(TYPE)>::type>;                                                \
    template class mo::TParam<mo::argument_type<void(TYPE)>::type>;                                                    \
    template struct mo::TPublisher<mo::argument_type<void(TYPE)>::type>

#define INSTANTIATE_META_PARAM(TYPE, TABLE)                                                                            \
    mo::MetaParam<TYPE, __COUNTER__> COMBINE(g_meta_Param, __LINE__)(TABLE, #TYPE);
