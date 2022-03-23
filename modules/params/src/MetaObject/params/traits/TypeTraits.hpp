#pragma once
#include "POD.hpp"
#include "Struct.hpp"
#include <MetaObject/logging/logging.hpp>
#include <boost/optional.hpp>
#include <memory>
#include <type_traits>

namespace rcc
{
    template <class T>
    struct weak_ptr;
    template <class T>
    struct shared_ptr;
} // namespace rcc

namespace mo
{
    // specializations for rcc::shared_ptr
    template <class Type>
    struct RccParamTraitsImplShared;
    template <class Type>
    struct RccParamTraitsImplWeak;
    template <class Type>
    struct ParamTraitsShared;

    // Base unspecialized forward declaration
    template <class Type, class Enable = void>
    struct ParamTraitsImpl
    {
    };

    // template<class Type, int N> struct TraitSelector{};

    // priority based selectors
    template <class Type, int N>
    struct TraitSelector : public TraitSelector<Type, N - 1>
    {
    };

    // High priority specializations for rcc::shared ptr types
    template <class Type>
    struct TraitSelector<rcc::shared_ptr<Type>, 1>
    {
        typedef RccParamTraitsImplShared<Type> TraitType;
    };

    template <class Type>
    struct TraitSelector<rcc::weak_ptr<Type>, 1>
    {
        typedef RccParamTraitsImplWeak<Type> TraitType;
    };

    // High priority specializations for std::shared_ptr
    // template<class Type> struct TraitSelector<std::shared_ptr<Type>, 1>{
    // typedef ParamTraitsShared<Type> TraitType;
    //};

    // Defaults to this with lowest priority
    template <class T>
    struct TraitSelector<T, 0>
    {
        typedef ParamTraitsImpl<T> TraitType;
    };

    // frontend, access everything through this
    template <class Type>
    struct ParamTraits : public TraitSelector<Type, 2>::TraitType
    {
    };
} // namespace mo
