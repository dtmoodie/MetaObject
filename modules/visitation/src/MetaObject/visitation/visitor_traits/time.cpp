#include "time.hpp"

namespace mo
{
    IWriteVisitor& visit(IWriteVisitor& visitor, mo::OptionalTime* time, const std::string&, const size_t)
    {
        bool set(*time);
        visitor(&set, "set");
        if(set)
        {
            auto sec = (*time)->seconds();
            visitor(&sec, "seconds");
        }
        return visitor;
    }
}
