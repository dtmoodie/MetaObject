#include "TraitRegistry.hpp"
#include <MetaObject/core/SystemTable.hpp>

namespace mo
{
    template <>
    struct ObjectConstructor<TraitRegistry>
    {
        using SharedPtr_t = std::shared_ptr<TraitRegistry>;
        using UniquePtr_t = std::unique_ptr<TraitRegistry>;

        SharedPtr_t createShared() const
        {
            return SharedPtr_t(new TraitRegistry());
        }

        UniquePtr_t createUnique() const
        {
            return UniquePtr_t(new TraitRegistry());
        }

        TraitRegistry* create() const
        {
            return new TraitRegistry();
        }
    };

    TraitRegistry& TraitRegistry::instance(SystemTable* table)
    {
        auto inst = table->getSingleton<TraitRegistry>();
        MO_ASSERT(inst);
        return *inst;
    }

    TraitRegistry& TraitRegistry::instance()
    {
        auto inst = singleton<TraitRegistry>();
        MO_ASSERT(inst);
        return *inst;
    }

    TraitRegistry::TraitRegistry()
    {
    }

    TraitRegistry::~TraitRegistry()
    {
    }

    void TraitRegistry::registerTrait(TypeInfo type, const ITraits* trait)
    {
        SystemTable::dispatchToSystemTable(
            [type, trait](SystemTable* table) { TraitRegistry::instance(table).registerTraitImpl(type, trait); });
    }

    void TraitRegistry::registerTraitImpl(TypeInfo type, const ITraits* trait)
    {
        auto itr = m_traits.find(type);
        if (itr == m_traits.end())
        {
            m_traits[type] = trait;
        }
    }

    std::unordered_map<TypeInfo, const ITraits*> TraitRegistry::getTraits()
    {
        return m_traits;
    }
} // namespace mo
