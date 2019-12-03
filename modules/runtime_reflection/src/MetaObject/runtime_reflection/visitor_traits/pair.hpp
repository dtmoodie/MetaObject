#ifndef MO_VISITATION_PAIR_HPP
#define MO_VISITATION_PAIR_HPP
#include "../IDynamicVisitor.hpp"
#include "../StructTraits.hpp"
#include "../type_traits.hpp"

#include <MetaObject/runtime_reflection/VisitorTraits.hpp>

#include <type_traits>

namespace mo
{
    /*template <class T1, class T2>
    struct TTraits<std::pair<T1, T2>, 9, void> : StructBase<std::pair<T1, T2>>
    {
        void load(ILoadVisitor& visitor, void* inst, const std::string&, size_t) const override
        {
            auto ptr = this->ptr(inst);
            visitor(&ptr->first, "first");
            visitor(&ptr->second, "second");

        }

        void save(ISaveVisitor& visitor, const void* inst, const std::string&, size_t) const override
        {
            auto ptr = this->ptr(inst);
            visitor(&ptr->first, "first");
            visitor(&ptr->second, "second");
        }

        void visit(StaticVisitor& visitor, const std::string&) const override
        {
            visitor.template visit<T1>("first");
            visitor.template visit<T2>("second");
        }
    };*/
}

namespace ct
{
    REFLECT_TEMPLATED_BEGIN(std::pair)
        PUBLIC_ACCESS(first)
        PUBLIC_ACCESS(second)
    REFLECT_END;
} // namespace ct

#endif // MO_VISITATION_PAIR_HPP
