#ifndef MO_VISITATION_TRAIT_REGISTRY_HPP
#define MO_VISITATION_TRAIT_REGISTRY_HPP

#include "TraitInterface.hpp"
#include <MetaObject/core/detail/ObjectConstructor.hpp>

#include <memory>
#include <unordered_map>

namespace mo
{
    struct ITraits;

    struct TraitRegistry
    {
        ~TraitRegistry();

        static TraitRegistry& instance();

        template <class TRAIT, class... Args>
        void registerTrait(const TypeInfo type, Args... args);

        const std::unordered_map<TypeInfo, std::unique_ptr<ITraits>>& getTraits();

      protected:
        friend ObjectConstructor<TraitRegistry>;
        TraitRegistry();

      private:
        std::unordered_map<TypeInfo, std::unique_ptr<ITraits>> m_traits;
    };

    template <class TRAIT, class... Args>
    void TraitRegistry::registerTrait(const TypeInfo type, Args... args)
    {
        m_traits[type] = std::unique_ptr<ITraits>(new TRAIT(args...));
    }

    template <class TYPE>
    struct TraitRegisterer
    {
        template <class... Args>
        TraitRegisterer(Args... args)
        {
            TraitRegistry::instance().template registerTrait<TTraits<TYPE, void>>(TypeInfo(typeid(TYPE)), args...);
        }
    };
}

#endif // MO_VISITATION_TRAIT_REGISTRY_HPP
