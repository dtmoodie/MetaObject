#include "HashVisitor.hpp"
#include <ct/Hash.hpp>

namespace mo
{

size_t HashVisitor::generateObjecthash(const IStructTraits *traits)
{
    m_hash = 0;
    traits->visit(this);
    return m_hash;
}

size_t HashVisitor::generateObjecthash(const IContainerTraits* traits)
{
    m_hash = 0;
    traits->visit(this);
    return m_hash;
}


void HashVisitor::implDyn(const TypeInfo type, const std::string& name, const size_t cnt)
{
    std::hash<std::string> hasher;
    m_hash = ct::combineHash<size_t>(m_hash, hasher(type.name()));
    m_hash = ct::combineHash<size_t>(m_hash, hasher(name));
}


}
