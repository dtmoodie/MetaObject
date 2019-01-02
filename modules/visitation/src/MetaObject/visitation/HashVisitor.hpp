#ifndef MO_VISITATION_HASH_VISITOR_HPP
#define MO_VISITATION_HASH_VISITOR_HPP
#include "IDynamicVisitor.hpp"

namespace mo
{
    struct HashVisitor: public StaticVisitor
    {
        size_t generateObjecthash(const IStructTraits* traits);
        size_t generateObjecthash(const IContainerTraits* traits);


    protected:
        virtual void implDyn(const TypeInfo, const std::string& name, const size_t cnt);
        size_t m_hash;
    };
}

#endif // MO_VISITATION_HASH_VISITOR_HPP
