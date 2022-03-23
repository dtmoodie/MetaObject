#include "HashVisitor.hpp"
#include <ct/hash.hpp>

namespace mo
{

    HashVisitor::HashVisitor(const bool hash_member_names, const bool hash_struct_names)
        : m_hash_member_names(hash_member_names)
        , m_hash_struct_names(hash_struct_names)
    {
    }

    size_t HashVisitor::generateObjecthash(const IStructTraits* traits, const std::string& name)
    {
        m_hash = 0;
        if (m_hash_struct_names)
        {
            auto type = traits->type();
            std::hash<std::string> hasher;
            m_hash = ct::combineHash<size_t>(m_hash, hasher(type.name()));
        }
        traits->visit(*this, name);
        return m_hash;
    }

    size_t HashVisitor::generateObjecthash(const IContainerTraits* traits, const std::string& name)
    {
        m_hash = 0;

        auto type = traits->type();
        std::hash<std::string> hasher;
        m_hash = ct::combineHash<size_t>(m_hash, hasher(type.name()));
        traits->visit(*this, name);
        return m_hash;
    }

    void HashVisitor::visit(const ITraits* trait, const std::string& name, const size_t cnt)
    {
        std::hash<std::string> hasher;
        if (m_hash_struct_names)
        {
            m_hash = ct::combineHash<size_t>(m_hash, hasher(trait->name()));
        }
        if (m_hash_member_names)
        {

            m_hash = ct::combineHash<size_t>(m_hash, hasher(name));
        }
        m_hash = ct::combineHash<size_t>(m_hash, cnt);
        trait->visit(*this, name);
    }

    void HashVisitor::implDyn(const TypeInfo type, const std::string& name, const size_t cnt)
    {
        std::hash<std::string> hasher;
        if (m_hash_member_names)
        {
            m_hash = ct::combineHash<size_t>(m_hash, hasher(name));
        }
        m_hash = ct::combineHash<size_t>(m_hash, hasher(type.name()));
        m_hash = ct::combineHash<size_t>(m_hash, cnt);
    }
} // namespace mo
