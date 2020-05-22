#ifndef MO_RUNTIME_REFLECTION_VISITOR_TRAITS_TYPEINFO_HPP
#define MO_RUNTIME_REFLECTION_VISITOR_TRAITS_TYPEINFO_HPP
#include "../StructTraits.hpp"
#include <MetaObject/detail/TypeInfo.hpp>
#include <MetaObject/runtime_reflection/IDynamicVisitor.hpp>
namespace mo
{
    template <>
    struct MO_EXPORTS TTraits<TypeInfo, 5> : StructBase<TypeInfo>
    {
        void load(ILoadVisitor& visitor, void* inst, const std::string&, size_t) const override;

        void save(ISaveVisitor& visitor, const void* inst, const std::string&, size_t) const override;

        void visit(StaticVisitor& visitor, const std::string&) const override;
    };
} // namespace mo
#endif // MO_RUNTIME_REFLECTION_VISITOR_TRAITS_TYPEINFO_HPP