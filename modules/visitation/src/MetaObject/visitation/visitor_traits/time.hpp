#ifndef MO_VISITOR_TRAITS_TIME_HPP
#define MO_VISITOR_TRAITS_TIME_HPP
#include <MetaObject/visitation/IDynamicVisitor.hpp>
#include <MetaObject/core/detail/Time.hpp>

namespace mo
{
    IWriteVisitor& visit(IWriteVisitor& visitor, mo::OptionalTime* time, const std::string&, const size_t);
}

#endif // MO_VISITOR_TRAITS_TIME_HPP
