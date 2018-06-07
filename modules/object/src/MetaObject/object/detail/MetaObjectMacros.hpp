#pragma once
#include "MetaObject/core/detail/NamedType.hpp"

namespace mo
{
    class ParamBase;
    class IParam;
    class ISlot;
    class ISignal;

    enum VisitationType
    {
        INPUT   = 0,
        OUTPUT  = 1,
        CONTROL = 2,
        STATUS  = 3,
        STATE   = 4,
        SIGNALS = 5,
        SLOTS   = 6
    };

    template<VisitationType Type>
    using VisitationFilter = std::integral_constant<VisitationType, Type>;

    using Name = NamedType<const char>;

    template<class T>
    using Param = NamedType<T, ParamBase>;

    template<class T>
    using Data = NamedType<T>;

    template<class T>
    Data<T> tagData(T* data)
    {
        return Data<T>(data);
    }

    template<class T>
    Param<T> tagParam(T& param )
    {
        return Param<T>(&param);
    }


    template<class T>
    using Slot = NamedType<T, ISlot>;

    template<class T>
    using Signal = NamedType<T, ISignal>;

    template<class T>
    Signal<T> tagSignal(T& sig)
    {
        return Signal<T>(&sig);
    }

    template<class T>
    Slot<T> tagSlot(T& sl)
    {
        return Slot<T>(&sl);
    }

    using Description = NamedType<const char>;

    using Tooltip = NamedType<const char>;

}

/*
   These two macros (MO_BEGIN kept for backwards compatibility) are used to define an
   interface base class.
*/
#define MO_BEGIN(CLASS_NAME) MO_BEGIN_1(CLASS_NAME, __COUNTER__)
#define MO_BASE(CLASS_NAME) MO_BEGIN_1(CLASS_NAME, __COUNTER__)

/*
    These two macros are used for defining a concrete class that has a valid implementation
*/
#define MO_DERIVE(CLASS_NAME, ...) MO_DERIVE_(__COUNTER__, CLASS_NAME, __VA_ARGS__)
#define MO_CONCRETE(CLASS_NAME, ...) MO_DERIVE_(__COUNTER__, CLASS_NAME, __VA_ARGS__)

/*
   This macro is used for defining a abstract class that derives from N interfaces without a
   concrete implementation
*/
#define MO_ABSTRACT(CLASS_NAME, ...) MO_ABSTRACT_(__COUNTER__, CLASS_NAME, __VA_ARGS__)

/*
    This macro is used for marking the end of a class definition block
*/
#define MO_END MO_END_(__COUNTER__)
#include "MetaObjectMacrosImpl.hpp"
