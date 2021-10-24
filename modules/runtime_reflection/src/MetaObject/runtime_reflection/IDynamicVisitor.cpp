#include "IDynamicVisitor.hpp"

namespace mo
{
    ILoadVisitor& ILoadVisitor::loadTrait(const IStructTraits* trait, void* inst, const std::string& name, size_t cnt)
    {
        return (*this)(trait, inst, name, cnt);
    }

    ILoadVisitor& ILoadVisitor::operator()(Byte* binary, const std::string& name, size_t num_bytes)
    {
        return (*this)(static_cast<void*>(binary), name, num_bytes);
    }

    ISaveVisitor& ISaveVisitor::operator()(const Byte* binary, const std::string& name, size_t bytes)
    {
        return (*this)(static_cast<const void*>(binary), name, bytes);
    }

    ILoadVisitor&
    ILoadVisitor::loadTrait(const IContainerTraits* trait, void* inst, const std::string& name, size_t cnt)
    {
        return (*this)(trait, inst, name, cnt);
    }

    ISaveVisitor&
    ISaveVisitor::saveTrait(const IStructTraits* trait, const void* inst, const std::string& name, size_t cnt)
    {
        return (*this)(trait, inst, name, cnt);
    }

    ISaveVisitor&
    ISaveVisitor::saveTrait(const IContainerTraits* trait, const void* inst, const std::string& name, size_t cnt)
    {
        return (*this)(trait, inst, name, cnt);
    }
} // namespace mo
