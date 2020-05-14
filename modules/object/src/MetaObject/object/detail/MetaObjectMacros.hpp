#ifndef MO_META_OBJECT_MACROS_HPP
#define MO_META_OBJECT_MACROS_HPP

#include "MetaObject/core/detail/NamedType.hpp"
#include <MetaObject/object/MetaObjectFactory.hpp>
#include <MetaObject/object/MetaObjectInfo.hpp>
#include <MetaObject/object/MetaObjectPolicy.hpp>

#include <ct/Indexer.hpp>
#include <ct/VariadicTypedef.hpp>
#include <ct/reflect.hpp>
#include <ct/reflect_macros.hpp>

#include <RuntimeObjectSystem/ObjectInterfacePerModule.h>
#include <RuntimeObjectSystem/shared_ptr.hpp>

#include <boost/preprocessor.hpp>

#include <string>
#include <type_traits>
#include <vector>

namespace mo
{
    template <class BASE>
    struct TParam;
    class IParam;
    class ISlot;
    class ISignal;

    enum VisitationMemberType
    {
        INPUT = 0,
        OUTPUT = 1,
        CONTROL = 2,
        STATUS = 3,
        STATE = 4,
        SIGNALS = 5,
        SLOTS = 6
    };

    enum VisitationType
    {
        LIST,
        TOOLTIP,
        INIT,
        SERIALIZE,
        DESCRIPTION
    };

    template <VisitationMemberType Type>
    using MemberFilter = std::integral_constant<VisitationMemberType, Type>;

    template <VisitationType Type>
    using VisitationFilter = std::integral_constant<VisitationType, Type>;

    using Name = NamedType<const char>;

    using Type = NamedType<const mo::TypeInfo>;

    template <class U>
    using Param = NamedType<TParam<U>>;

    template <class T>
    using Data = NamedType<T>;

    template <class T>
    Data<T> tagData(T* data)
    {
        return Data<T>(data);
    }

    template <class T>
    using Slot = NamedType<T, ISlot>;

    template <class T>
    using Signal = NamedType<T, ISignal>;

    template <class T>
    Signal<T> tagSignal(T& sig)
    {
        return Signal<T>(&sig);
    }

    template <class T>
    Slot<T> tagSlot(T& sl)
    {
        return Slot<T>(&sl);
    }

    template <class T>
    Type tagType()
    {
        return Type(typeid(T));
    }

    template <class... T>
    std::vector<Type> tagTypePack()
    {
        return {Type(typeid(T))...};
    }

    template <class T, class R, class... Args>
    NamedType<R (T::*)(Args...), Function> tagFunction(R (T::*ptr)(Args...))
    {
        return NamedType<R (T::*)(Args...), Function>(ptr);
    }

    template <class R, class... Args>
    NamedType<R (*)(Args...), StaticFunction> tagStaticFunction(R (*ptr)(Args...))
    {
        return NamedType<R (*)(Args...), StaticFunction>(ptr);
    }

    using Description = NamedType<const char>;

    using Tooltip = NamedType<const char>;
} // namespace mo

/*
   These two macros (MO_BEGIN kept for backwards compatibility) are used to define an
   interface base class.
*/
#define MO_BEGIN(TYPE)                                                                                                 \
    REFLECT_INTERNAL_BEGIN(TYPE)                                                                                       \
        static rcc::shared_ptr<DataType> create();

#define MO_BASE(TYPE) REFLECT_INTERNAL_BEGIN(TYPE)

/*
    These two macros are used for defining a concrete class that has a valid implementation
*/
#define MO_DERIVE(TYPE, ...)                                                                                           \
    REFLECT_INTERNAL_DERIVED(TYPE, __VA_ARGS__)                                                                        \
    static rcc::shared_ptr<DataType> create();                                                                         \
    static constexpr ct::StringView getTypeName()                                                                      \
    {                                                                                                                  \
        return #TYPE;                                                                                                  \
    }

#define MO_CONCRETE(...) MO_DERIVE(__VA_ARGS__)

/*
   This macro is used for defining a abstract class that derives from N interfaces without a
   concrete implementation
*/
#define MO_ABSTRACT(TYPE, ...)                                                                                         \
    REFLECT_INTERNAL_DERIVED(TYPE, __VA_ARGS__)                                                                        \
    static constexpr ct::StringView getTypeName()                                                                      \
    {                                                                                                                  \
        return #TYPE;                                                                                                  \
    }

/*
    This macro is used for marking the end of a class definition block
*/
#define MO_END REFLECT_INTERNAL_END

struct ISimpleSerializer;
namespace mo
{
    struct SignalInfo;
    struct SlotInfo;
    struct ParamInfo;
} // namespace mo

#define REFLECT_START(N_)                                                                                              \
    template <class V, ct::index_t N, class F, class T, class... Args>                                                 \
    inline void reflectHelper(V& visitor, F filter, T type, const ct::Indexer<N> dummy, Args&&... args)                \
    {                                                                                                                  \
        reflectHelper(visitor, filter, type, --dummy, args...);                                                        \
    }                                                                                                                  \
    template <class V, class F, class T, class... Args>                                                                \
    inline void reflectHelper(V&, F, T, const ct::Indexer<N_>, Args&&...)                                              \
    {                                                                                                                  \
    }                                                                                                                  \
    template <class V, ct::index_t N, class F, class T, class... Args>                                                 \
    static inline void reflectHelperStatic(V& visitor, F filter, T type, const ct::Indexer<N> dummy, Args&&... args)   \
    {                                                                                                                  \
        reflectHelperStatic(visitor, filter, type, --dummy, args...);                                                  \
    }                                                                                                                  \
    template <class V, class F, class T, class... Args>                                                                \
    static inline void reflectHelperStatic(V&, F, T, const ct::Indexer<N_>, Args&&...)                                 \
    {                                                                                                                  \
    }

#define MO_BEGIN_1(CLASS_NAME, N_)                                                                                     \
    using THIS_CLASS = CLASS_NAME;                                                                                     \
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

#define MO_ABSTRACT_(N_, CLASS_NAME, ...)                                                                              \
    using THIS_CLASS = CLASS_NAME;                                                                                     \
    REFLECT_START(N_)

#define MO_REGISTER_OBJECT(TYPE)                                                                                       \
    static ::mo::MetaObjectInfo<TActual<TYPE>> TYPE##_info;                                                            \
    static ::mo::MetaObjectPolicy<TActual<TYPE>, __COUNTER__, void> TYPE##_policy;                                     \
    ::rcc::shared_ptr<TYPE> TYPE::create()                                                                             \
    {                                                                                                                  \
        auto obj = ::mo::MetaObjectFactory::instance()->create(#TYPE);                                                 \
        return ::rcc::shared_ptr<TYPE>(obj);                                                                           \
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

#endif // MO_META_OBJECT_MACROS_HPP
