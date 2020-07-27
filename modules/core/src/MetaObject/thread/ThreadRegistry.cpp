#include "ThreadRegistry.hpp"

#include <algorithm>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>
using namespace mo;

struct ThreadRegistry::impl
{
    std::map<int, std::vector<IAsyncStreamPtr_t>> _thread_map;
    std::mutex mtx;
};

using namespace mo;

size_t tid()
{
    std::stringstream ss;
    ss << std::this_thread::get_id();
    size_t output;
    ss >> output;
    return output;
}

size_t mo::getThisThread()
{
    thread_local size_t thread_id = tid();
    return thread_id;
}

ThreadRegistry::ThreadRegistry()
{
    _pimpl = new impl();
}

ThreadRegistry::~ThreadRegistry()
{
    delete _pimpl;
}

void ThreadRegistry::registerThread(ThreadType type, mo::IAsyncStreamPtr_t stream)
{
    std::lock_guard<std::mutex> lock(_pimpl->mtx);
    auto& threads = _pimpl->_thread_map[type];
    if (std::count(threads.begin(), threads.end(), stream) == 0)
    {
        threads.push_back(stream);
    }
}

mo::IAsyncStreamPtr_t ThreadRegistry::getThread(ThreadType type_)
{
    const int type = type_;
    std::lock_guard<std::mutex> lock(_pimpl->mtx);
    // TODO some kind of load balancing for multiple threads of a specific type
    IAsyncStreamPtr_t current_thread = IAsyncStream::current();
    auto itr = _pimpl->_thread_map.find(type);
    if (itr != _pimpl->_thread_map.end())
    {
        if (!itr->second.empty())
        {
            if (std::count(itr->second.begin(), itr->second.end(), current_thread) == 0)
            {
                return itr->second.back();
            }

            return current_thread;
        }
    }
    return {};
}

ThreadRegistry* ThreadRegistry::instance()
{
    static ThreadRegistry* inst = nullptr;
    if (inst == nullptr)
        inst = new ThreadRegistry();
    return inst;
}
