#pragma once
#include "MetaObject/core/detail/Counter.hpp"
#include "MetaObject/object/MetaObjectFactory.hpp"
#include "MetaObject/object/MetaObjectInfo.hpp"
#include "MetaObject/object/MetaObjectPolicy.hpp"
#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"
#include "ct/VariadicTypedef.hpp"
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
    template <class V, int N, class F, class T, class... Args>                                                         \
    inline void reflectHelper(V& visitor, F filter, T type, mo::_counter_<N> dummy, Args&&... args)                    \
    {                                                                                                                  \
        reflectHelper(visitor, filter, type, --dummy, args...);                                                        \
    }                                                                                                                  \
    template <class V, class F, class T, class... Args>                                                                \
    inline void reflectHelper(V&, F, T, mo::_counter_<N_>, Args&&...)                                                  \
    {                                                                                                                  \
    }                                                                                                                  \
    template <class V, int N, class F, class T, class... Args>                                                         \
    static inline void reflectHelperStatic(V& visitor, F filter, T type, mo::_counter_<N> dummy, Args&&... args)       \
    {                                                                                                                  \
        reflectHelperStatic(visitor, filter, type, --dummy, args...);                                                  \
    }                                                                                                                  \
    template <class V, class F, class T, class... Args>                                                                \
    static inline void reflectHelperStatic(V&, F, T, mo::_counter_<N_>, Args&&...)                                     \
    {                                                                                                                  \
    }

#define MO_BEGIN_1(CLASS_NAME, N_)                                                                                     \
    using THIS_CLASS = CLASS_NAME;                                                                                     \
    using ParentClass = ct::VariadicTypedef<void>;                                                                     \
    REFLECT_START(N_)                                                                                                  \
    static rcc::shared_ptr<THIS_CLASS::InterfaceHelper<CLASS_NAME>> create();

#define MO_DERIVE_(N_, CLASS_NAME, ...)                                                                                \
    using THIS_CLASS = CLASS_NAME;                                                                                     \
    using ParentClass = ct::VariadicTypedef<__VA_ARGS__>;                                                              \
    REFLECT_START(N_)                                                                                                  \
    static rcc::shared_ptr<THIS_CLASS::InterfaceHelper<CLASS_NAME>> create();

template <class T>
struct ReflectParent;

template <>
struct ReflectParent<ct::VariadicTypedef<void>>
{
    template <class T, class Visitor, class Filter, class Type, class... Args>
    static void visit(T*, Visitor&, Filter, Type, Args&&...)
    {
    }

    template <class Visitor, class Filter, class Type, class... Args>
    static void visit(Visitor&, Filter, Type, Args&&...)
    {
    }
};

template <class Parent>
struct ReflectParent<ct::VariadicTypedef<Parent>>
{
    template <class T, class Visitor, class Filter, class Type, class... Args>
    static void visit(T* obj, Visitor& visitor, Filter filter, Type type, Args&&... args)
    {
        obj->Parent::reflect(visitor, filter, type, std::forward<Args>(args)...);
    }

    template <class Visitor, class Filter, class Type, class... Args>
    static void visit(Visitor& visitor, Filter filter, Type type, Args&&... args)
    {
        Parent::reflectStatic(visitor, filter, type, std::forward<Args>(args)...);
    }
};

template <class Parent, class... Parents>
struct ReflectParent<ct::VariadicTypedef<Parent, Parents...>>
{
    template <class T, class Visitor, class Filter, class Type, class... Args>
    static void visit(T* obj, Visitor& visitor, Filter filter, Type type, Args&&... args)
    {
        obj->Parent::reflect(visitor, filter, type, std::forward<Args>(args)...);
        ReflectParent<ct::VariadicTypedef<Parents...>>::visit(obj, visitor, filter, type, std::forward<Args>(args)...);
    }

    template <class Visitor, class Filter, class Type, class... Args>
    static void visit(Visitor& visitor, Filter filter, Type type, Args&&... args)
    {
        Parent::reflectStatic(visitor, filter, type, std::forward<Args>(args)...);
        ReflectParent<ct::VariadicTypedef<Parents...>>::visit(visitor, filter, type, std::forward<Args>(args)...);
    }
};

#define MO_END_(N)                                                                                                     \
    template <class V, class F, class T, class... Args>                                                                \
    inline void reflect(V& visitor, F filter, T type, Args&&... args)                                                  \
    {                                                                                                                  \
        ReflectParent<ParentClass>::visit(this, visitor, filter, type, std::forward<Args>(args)...);                   \
        reflectHelper(visitor, filter, type, mo::_counter_<N>(), std::forward<Args>(args)...);                         \
    }                                                                                                                  \
    template <class V, class F, class T, class... Args>                                                                \
    static inline void reflectStatic(V& visitor, F filter, T type, Args&&... args)                                     \
    {                                                                                                                  \
        ReflectParent<ParentClass>::visit(visitor, filter, type, std::forward<Args>(args)...);                         \
        reflectHelperStatic(visitor, filter, type, mo::_counter_<N>(), std::forward<Args>(args)...);                   \
    }

#define MO_ABSTRACT_(N_, CLASS_NAME, ...)                                                                              \
    using THIS_CLASS = CLASS_NAME;                                                                                     \
    using ParentClass = ct::VariadicTypedef<__VA_ARGS__>;                                                              \
    REFLECT_START(N_)

#define MO_REGISTER_OBJECT(TYPE)                                                                                       \
    static ::mo::MetaObjectInfo<TActual<TYPE>> TYPE##_info;                                                            \
    static ::mo::MetaObjectPolicy<TActual<TYPE>, __COUNTER__, void> TYPE##_policy;                                     \
    ::rcc::shared_ptr<TYPE::InterfaceHelper<TYPE>> TYPE::create()                                                      \
    {                                                                                                                  \
        auto obj = ::mo::MetaObjectFactory::instance()->create(#TYPE);                                                 \
        return ::rcc::shared_ptr<InterfaceHelper<TYPE>>(obj);                                                          \
    }                                                                                                                  \
    REGISTERCLASS(TYPE, &TYPE##_info);

#define MO_REGISTER_CLASS(TYPE) MO_REGISTER_OBJECT(TYPE)


#define MO_OBJ_TOOLTIP(tooltip)                                                                                        \
    static std::string getTooltipStatic()                                                                              \
    {                                                                                                                  \
        return tooltip;                                                                                                \
    }

#define MO_OBJ_DESCRIPTION(desc)                                                                                       \
    static std::string getDescriptionStatic()                                                                          \
    {                                                                                                                  \
        return desc;                                                                                                   \
    }
