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
        auto inst = singleton<TraitRegistry>(table);
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

    const std::unordered_map<TypeInfo, std::unique_ptr<ITraits>>& TraitRegistry::getTraits()
    {
        return m_traits;
    }
}
