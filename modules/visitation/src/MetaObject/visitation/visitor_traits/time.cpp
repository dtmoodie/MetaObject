#include "time.hpp"

namespace mo
{
    template <>
    IWriteVisitor& visit(IWriteVisitor& visitor, const mo::OptionalTime* time, const std::string&, const size_t)
    {
        const bool set(*time);
        visitor(&set, "set");
        if (set)
        {
            auto sec = (*time)->seconds();
            visitor(&sec, "seconds");
        }
        return visitor;
    }
}
