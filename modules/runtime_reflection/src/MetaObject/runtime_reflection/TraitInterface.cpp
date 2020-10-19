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

    std::string IStructTraits::getMemberName(uint32_t idx) const
    {
        return {};
    }

    int32_t IStructTraits::getMemberIndex(const std::string& name) const
    {
        return -1;
    }

    bool IStructTraits::loadMember(ILoadVisitor&, void*, uint32_t, std::string*) const
    {
        return false;
    }

    bool IStructTraits::saveMember(ISaveVisitor&, const void*, uint32_t, std::string*) const
    {
        return false;
    }

    bool IStructTraits::loadMember(ILoadVisitor& visitor, void* inst, const std::string& name, uint32_t* idx) const
    {
        return false;
    }

    bool
    IStructTraits::saveMember(ISaveVisitor& visitor, const void* inst, const std::string& name, uint32_t* idx) const
    {
        return false;
    }
} // namespace mo
