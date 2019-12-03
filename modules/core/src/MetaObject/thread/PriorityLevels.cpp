#include "PriorityLevels.hpp"

namespace std
{
    ostream& operator<<(ostream& os, const mo::PriorityLevels p)
    {
        switch (p)
        {
        case mo::NONE:
            return (os << "PRIORITY:NONE");
        case mo::LOWEST:
            return (os << "PRIORITY:LOWEST");
        case mo::LOW:
            return (os << "PRIORITY:LOW");
        case mo::MEDIUM:
            return (os << "PRIORITY:MEDIUM");
        case mo::HIGH:
            return (os << "PRIORITY:HIGH");
        case mo::HIGHEST:
            return (os << "PRIORITY:HIGHEST");
        default:
            return os;
        }
        return os;
    }
}
