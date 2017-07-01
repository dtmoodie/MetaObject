#pragma once
#include <cstddef>
#include "MetaObject/detail/Export.hpp"
#include <string>
namespace boost
{
    class thread;
}
namespace mo
{
    size_t MO_EXPORTS getThreadId(const boost::thread& thread);
    void MO_EXPORTS setThreadName(boost::thread& thread, const std::string& name);
}
