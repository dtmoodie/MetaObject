#ifndef MO_VISITOR_TRAITS_TIME_HPP
#define MO_VISITOR_TRAITS_TIME_HPP
#include <MetaObject/core/detail/Time.hpp>
#include <MetaObject/runtime_reflection/IDynamicVisitor.hpp>

namespace mo
{
    template<>
    struct Visit<mo::OptionalTime>
    {
        static ILoadVisitor& load(ILoadVisitor& visitor, mo::OptionalTime* time, const std::string&, const size_t);
        static ISaveVisitor& save(ISaveVisitor& visitor, const mo::OptionalTime* time, const std::string&, const size_t);
        static void visit(StaticVisitor& visitor, const std::string& name, const size_t cnt);
    };

}

#endif // MO_VISITOR_TRAITS_TIME_HPP
