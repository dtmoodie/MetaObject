#ifndef MO_THREAD_PRIORITY_LEVELS_HPP
#define MO_THREAD_PRIORITY_LEVELS_HPP
#include <MetaObject/core/detail/forward.hpp>

namespace mo
{
    enum PriorityLevels
    {
        NONE = -1,
        LOWEST = 0,
        LOW,
        MEDIUM,
        DEFAULT = MEDIUM,
        HIGH,
        HIGHEST
    };
}

namespace std
{
    ostream& operator<<(ostream&, mo::PriorityLevels);
}

#endif // MO_THREAD_PRIORITY_LEVELS_HPP
