#ifndef MO_THREAD_THREAD_INFO_HPP
#define MO_THREAD_THREAD_INFO_HPP
#include <MetaObject/core.hpp>

namespace boost
{
    class thread;
}
namespace std
{
    class thread;
}
namespace mo
{
    size_t MO_EXPORTS getThisThreadId();
    std::string MO_EXPORTS getThisThreadName();
    void MO_EXPORTS setThisThreadName(const std::string& name);

    void MO_EXPORTS setThreadName(boost::thread& thread, const std::string& name);

    size_t MO_EXPORTS getThreadId(const boost::thread& thread);
    std::string MO_EXPORTS getThreadName(boost::thread& thread);

} // namespace mo

#endif // MO_THREAD_THREAD_INFO_HPP