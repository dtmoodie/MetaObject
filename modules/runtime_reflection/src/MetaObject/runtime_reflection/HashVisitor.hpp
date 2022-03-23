#ifndef MO_VISITATION_HASH_VISITOR_HPP
#define MO_VISITATION_HASH_VISITOR_HPP
#include "IDynamicVisitor.hpp"

namespace mo
{
    struct MO_EXPORTS HashVisitor : public StaticVisitor
    {
        HashVisitor(const bool hash_member_names = true, const bool hash_struct_names = true);

        size_t generateObjecthash(const IStructTraits* traits, const std::string& name = "");
        size_t generateObjecthash(const IContainerTraits* traits, const std::string& name = "");

        void visit(const ITraits*, const std::string& name, const size_t cnt = 1) override;

      protected:
        void implDyn(const TypeInfo, const std::string& name, const size_t cnt) override;
        size_t m_hash;
        bool m_hash_member_names;
        bool m_hash_struct_names;
    };
} // namespace mo

#endif // MO_VISITATION_HASH_VISITOR_HPP
