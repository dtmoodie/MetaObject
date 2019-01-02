#include "time.hpp"

namespace mo
{
    IReadVisitor& Visit<OptionalTime>::read(IReadVisitor& visitor, mo::OptionalTime* time, const std::string&, const size_t)
    {
        bool set = false;
        visitor(&set, "set");
        if(set)
        {
            double sec;
            visitor(&sec, "seconds");
            (**time) = Time(sec);
        }
        return visitor;
    }

    IWriteVisitor& Visit<OptionalTime>::write(IWriteVisitor& visitor, const mo::OptionalTime* time, const std::string&, const size_t)
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

    void Visit<OptionalTime>::visit(StaticVisitor& visitor, const std::string& , const size_t )
    {
        visitor.visit<bool>("set");
        visitor.visit<double>("seconds");
    }
}
