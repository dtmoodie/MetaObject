#include "TraitInterface.hpp"
#include "TraitRegistry.hpp"
namespace mo
{

    std::string ITraits::name() const
    {
        const auto t = type();
        return t.name();
    }

    ITraits::~ITraits() = default;

    uint32_t IStructTraits::getNumMembers() const
    {
        return 0;
    }

    bool IStructTraits::getMember(
        void* inst, void** member, const IStructTraits** trait, uint32_t idx, std::string* name) const
    {
        return false;
    }

    bool IStructTraits::getMember(
        const void* inst, const void** member, const IStructTraits** trait, uint32_t idx, std::string* name) const
    {
        return false;
    }

} // namespace mo
