#include <cstddef>
#include "MetaObject/thread/BoostThread.hpp"
#include "boost/thread.hpp"
#ifdef _MSC_VER

#else
#include <pthread.h>
#endif

size_t mo::getThreadId(const boost::thread& thread)
{
    std::stringstream ss;
    ss << std::hex << thread.get_id();
    size_t output;
    ss >> output;
    return output;
}
void mo::setThreadName(boost::thread& thread, const std::string& name){
#ifdef _MSC_VER
    
#else
    pthread_t tid = static_cast<pthread_t>(thread.native_handle());
    pthread_setname_np(tid, name.c_str());
#endif
}
