#ifndef MO_VISITOR_TRAITS_TIME_HPP
#define MO_VISITOR_TRAITS_TIME_HPP
#include <MetaObject/core/detail/Time.hpp>
#include <MetaObject/visitation/IDynamicVisitor.hpp>

namespace mo
{
    IReadVisitor& read(IReadVisitor& visitor, mo::OptionalTime* time, const std::string&, const size_t);
    IWriteVisitor& write(IWriteVisitor& visitor, const mo::OptionalTime* time, const std::string&, const size_t);
}

#endif // MO_VISITOR_TRAITS_TIME_HPP
