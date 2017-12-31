#pragma once
#include "MetaObject/detail/Export.hpp"
#include <cstddef>

namespace std
{
    class thread;
}

namespace mo
{
    size_t MO_EXPORTS getThreadId(const std::thread& thread);
}
