#pragma once
#include <cstddef>
#include "MetaObject/detail/Export.hpp"

namespace boost
{
    class thread;
}
namespace mo
{
    size_t MO_EXPORTS getThreadId(const boost::thread& thread);
}
