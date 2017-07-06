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
    void MO_EXPORTS setThisThreadName(const std::string& name);
    std::string MO_EXPORTS getThreadName(boost::thread& thread);
    std::string MO_EXPORTS getThisThreadName();
}
