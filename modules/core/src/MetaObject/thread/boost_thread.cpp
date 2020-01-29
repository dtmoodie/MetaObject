#include "MetaObject/thread/boost_thread.hpp"
#include "MetaObject/logging/logging.hpp"
#include <boost/optional.hpp>
#include <boost/thread.hpp>
#include <cstddef>
#ifdef _MSC_VER

#else
#include <pthread.h>
#endif

size_t getThreadIdHelper(const boost::thread& thread)
{
    std::stringstream ss;
    ss << std::hex << thread.get_id();
    size_t output;
    ss >> output;
    return output;
}

size_t mo::getThreadId(const boost::thread& thread)
{
    thread_local boost::optional<size_t> tid;
    if (!tid)
        tid = getThreadIdHelper(thread);
    return *tid;
}

void mo::setThreadName(boost::thread& thread, const std::string& name)
{
#ifdef _MSC_VER

#else
    pthread_t tid = static_cast<pthread_t>(thread.native_handle());
    if (pthread_setname_np(tid, name.c_str()) != 0)
    {
        MO_LOG(info, "Failed to set thread [{}] to name '{}'", tid, name);
    }
#endif
}

void mo::setThisThreadName(const std::string& name)
{
#ifdef _MSC_VER

#else
    pthread_t tid = pthread_self();
    if (name.size() > 16)
    {
        if (pthread_setname_np(tid, name.substr(0, 15).c_str()) != 0)
        {
            MO_LOG(info, "Failed to set thread [{}] to name '{}'", tid, name);
        }
    }
    else
    {
        if (pthread_setname_np(tid, name.c_str()) != 0)
        {
            MO_LOG(info, "Failed to set thread [{}] to name '{}'", tid, name);
        }
    }

#endif
}

std::string mo::getThreadName(boost::thread& thread)
{
    std::string output;
#ifdef _MSC_VER

#else
    pthread_t tid = static_cast<pthread_t>(thread.native_handle());
    thread_local char buf[100] = {'\0'};
    if (pthread_getname_np(tid, buf, 100) != 0)
    {
        MO_LOG(info, "Failed to get thread [{}] name ", tid);
    }
    output = std::string(buf);
#endif
    return output;
}

std::string mo::getThisThreadName()
{
    std::string output;
#ifdef _MSC_VER

#else
    pthread_t tid = pthread_self();
    thread_local char buf[100] = {'\0'};
    if (pthread_getname_np(tid, buf, 100) != 0)
    {
        MO_LOG(info, "Failed to get thread [{}] name ", tid);
    }
    output = std::string(buf);
#endif
    return output;
}
