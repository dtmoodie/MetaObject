#include "TypeInfo.hpp"
#include <MetaObject/core/TypeTable.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/string.hpp>

namespace mo
{

    void TTraits<TypeInfo, 5>::load(ILoadVisitor& visitor, void* inst, const std::string& name, size_t) const
    {
        auto& ref = this->ref(inst);
        std::string type_name;
        visitor(&type_name, name);
        ref = TypeTable::instance()->nameToType(type_name);
    }

    void TTraits<TypeInfo, 5>::save(ISaveVisitor& visitor, const void* inst, const std::string& name, size_t) const
    {
        const auto& ref = this->ref(inst);
        auto type_table = TypeTable::instance();
        const auto type_name = type_table->typeToName(ref);
        visitor(&type_name, name);
    }

    void TTraits<TypeInfo, 5>::visit(StaticVisitor& visitor, const std::string&) const
    {
        visitor.template visit<std::string>("name");
    }
} // namespace mo