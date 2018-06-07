#pragma once
#ifndef __CUDACC__
#include "MetaObject/core/detail/Counter.hpp"
#include "MetaObject/object/MetaObjectFactory.hpp"
#include "MetaObject/object/MetaObjectInfo.hpp"
#include "MetaObject/object/MetaObjectPolicy.hpp"
#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"
#include <RuntimeObjectSystem/shared_ptr.hpp>
#include <boost/preprocessor.hpp>
#include <string>
#include <type_traits>
#include <vector>

struct ISimpleSerializer;
namespace mo
{
    struct SignalInfo;
    struct SlotInfo;
    struct ParamInfo;
}

#define REFLECT_START(N_)                                                                                              \
template<class V, int N, class T, class ... Args>                                                                      \
static constexpr inline void reflectHelper(V& visitor, THIS_CLASS* obj, T type, mo::_counter_<N> dummy, Args&&... args) {    \
    reflectHelper(visitor, obj, type, --dummy, args...);                                                                     \
}                                                                                                                      \
template<class V, class T, class... Args>                                                                              \
static constexpr inline void reflectHelper(V& visitor, THIS_CLASS* obj, T , mo::_counter_<N_>, Args&&...) {}

#define MO_BEGIN_1(CLASS_NAME, N_) \
    using THIS_CLASS = CLASS_NAME; \
    using ParentClass = std::tuple<void>; \
    REFLECT_START(N_) \
    static rcc::shared_ptr<CLASS_NAME> create();


#define MO_DERIVE_(N_, CLASS_NAME, ...)                                                                                \
    using THIS_CLASS = CLASS_NAME; \
    using ParentClass = std::tuple<__VA_ARGS__>; \
    REFLECT_START(N_) \
    static rcc::shared_ptr<CLASS_NAME> create();

#define MO_END_(N) \
template<class V, class T, class ... Args> \
static constexpr inline void reflect(V& visitor, THIS_CLASS* obj, T filter, Args&&... args) \
{ \
    reflectHelper(visitor, obj, filter, mo::_counter_<N>(), std::forward<Args>(args)...); \
}

#define MO_ABSTRACT_(N_, CLASS_NAME, ...)                                                                              \
    using THIS_CLASS = CLASS_NAME; \
    using ParentClass = std::tuple<__VA_ARGS__>; \
    REFLECT_START(N_)

#define MO_REGISTER_OBJECT(TYPE)                                                                                       \
    static ::mo::MetaObjectInfo<TActual<TYPE>> TYPE##_info;                                                            \
    static ::mo::MetaObjectPolicy<TActual<TYPE>, __COUNTER__, void> TYPE##_policy;                                     \
    ::rcc::shared_ptr<TYPE> TYPE::create()                                                                             \
    {                                                                                                                  \
        auto obj = ::mo::MetaObjectFactory::instance().create(#TYPE);                                                 \
        return ::rcc::shared_ptr<TYPE>(obj);                                                                           \
    }                                                                                                                  \
    REGISTERCLASS(TYPE, &TYPE##_info);

#define MO_REGISTER_CLASS(TYPE) MO_REGISTER_OBJECT(TYPE)

#else // __CUDACC__
#define MO_REGISTER_OBJECT(TYPE)
#define MO_REGISTER_CLASS(TYPE)
#define MO_BEGIN_1(CLASS, N)
#define MO_BEGIN_2(CLASS, PARENT, N)
#define MO_END_(N)
#endif // __CUDACC__
