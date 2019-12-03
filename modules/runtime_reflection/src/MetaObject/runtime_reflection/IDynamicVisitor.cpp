#include "IDynamicVisitor.hpp"

namespace mo
{
    ILoadVisitor& ILoadVisitor::loadTrait(IStructTraits* trait, void* inst, const std::string& name, size_t cnt)
    {
        return (*this)(trait, inst, name, cnt);
    }

    ILoadVisitor& ILoadVisitor::loadTrait(IContainerTraits* trait, void* inst, const std::string& name, size_t cnt)
    {
        return (*this)(trait, inst, name, cnt);
    }

    ISaveVisitor& ISaveVisitor::saveTrait(IStructTraits* trait, const void* inst, const std::string& name, size_t cnt)
    {
        return (*this)(trait, inst, name, cnt);
    }

    ISaveVisitor& ISaveVisitor::saveTrait(IContainerTraits* trait, const void* inst, const std::string& name, size_t cnt)
    {
        return (*this)(trait, inst, name, cnt);
    }
}
