#include "std_thread.hpp"
#include <sstream>
#include <thread>

size_t mo::getThreadId(const std::thread& thread)
{
    std::stringstream ss;
    ss << thread.get_id();
    size_t output;
    ss >> output;
    return output;
}
