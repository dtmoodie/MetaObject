#pragma once
#include <boost/optional.hpp>
#include <type_traits>
#include <memory>
#include <MetaObject/logging/Log.hpp>
#include "POD.hpp"
#include "Struct.hpp"

namespace rcc{
    template<class T> class weak_ptr;
    template<class T> class shared_ptr;
}

namespace mo {
// specializations for rcc::shared_ptr
template<class Type> struct RccParamTraitsImplShared;
template<class Type> struct RccParamTraitsImplWeak;
template<class Type> struct ParamTraitsShared;

// Base unspecialized forward declaration
template<class Type, class Enable = void> struct ParamTraitsImpl {};


// priority based selectors
template<class Type, int Priority> struct TraitSelector: public TraitSelector<Type, Priority - 1>{};

// High priority specializations for rcc::shared ptr types
template<class Type> struct TraitSelector<rcc::shared_ptr<Type>, 1>{
    typedef RccParamTraitsImplShared<Type> TraitType;
};

template<class Type> struct TraitSelector<rcc::weak_ptr<Type>, 1>{
    typedef RccParamTraitsImplWeak<Type> TraitType;
};

// High priority specializations for std::shared_ptr
template<class Type> struct TraitSelector<std::shared_ptr<Type>, 1>{
    typedef ParamTraitsShared<Type> TraitType;
};

// Defaults to this with lowest priority
template<class T> struct TraitSelector<T, 0>{
    typedef ParamTraitsImpl<T> TraitType;
};

// frontend, access everything through this
template<class Type> struct ParamTraits: public TraitSelector<Type, 1>::TraitType{};

}
