#include <cstddef>
#include "MetaObject/Thread/BoostThread.hpp"

#include "boost/thread.hpp"


size_t mo::GetThreadId(const boost::thread& thread)
{
    std::stringstream ss;
    ss << std::hex << thread.get_id();
    size_t output;
    ss >> output;
    return output;
}
