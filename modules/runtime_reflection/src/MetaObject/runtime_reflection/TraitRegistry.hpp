#ifndef MO_VISITATION_TRAIT_REGISTRY_HPP
#define MO_VISITATION_TRAIT_REGISTRY_HPP
#include <MetaObject/core/detail/ObjectConstructor.hpp>
#include <MetaObject/detail/TypeInfo.hpp>

#include <unordered_map>

struct SystemTable;

namespace mo
{
    struct ITraits;

    struct MO_EXPORTS TraitRegistry
    {
        ~TraitRegistry();

        static TraitRegistry& instance();
        static TraitRegistry& instance(SystemTable*);

        static void registerTrait(TypeInfo type, const ITraits*);

        std::unordered_map<TypeInfo, const ITraits*> getTraits();

      protected:
        friend ObjectConstructor<TraitRegistry>;
        TraitRegistry();

      private:
        void registerTraitImpl(TypeInfo type, const ITraits*);
        std::unordered_map<TypeInfo, const ITraits*> m_traits;
    };
} // namespace mo

#endif // MO_VISITATION_TRAIT_REGISTRY_HPP
