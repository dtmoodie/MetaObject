#ifndef MO_META_OBJECT_MACROS_HPP
#define MO_META_OBJECT_MACROS_HPP
#include "MetaObject/core/detail/NamedType.hpp"
#include <ct/reflect.hpp>
#include <ct/reflect_macros.hpp>
#include <vector>

namespace mo
{
    class ParamBase;
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

    template <class T>
    using Param = NamedType<T, ParamBase>;

    template <class T>
    using Data = NamedType<T>;

    template <class T>
    Data<T> tagData(T* data)
    {
        return Data<T>(data);
    }

    template <class T>
    Param<T> tagParam(T& param)
    {
        return Param<T>(&param);
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
    static constexpr const ct::index_t REFLECT_COUNT_BEGIN = __COUNTER__ + 1;                                          \
    using DataType = TYPE;                                                                                             \
    using BaseTypes = ct::VariadicTypedef<__VA_ARGS__>;                                                                \
    using ParentClass = ct::VariadicTypedef<__VA_ARGS__>;                                                              \
    static rcc::shared_ptr<DataType> create();

#define MO_CONCRETE(...) MO_DERIVE(__VA_ARGS__)

/*
   This macro is used for defining a abstract class that derives from N interfaces without a
   concrete implementation
*/
//#define MO_ABSTRACT(CLASS_NAME, ...) MO_ABSTRACT_(__COUNTER__, CLASS_NAME, __VA_ARGS__)

/*
    This macro is used for marking the end of a class definition block
*/
#define MO_END REFLECT_INTERNAL_END

#include "MetaObjectMacrosImpl.hpp"
#endif // MO_META_OBJECT_MACROS_HPP
