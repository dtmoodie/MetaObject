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
    std::map<int, std::vector<std::weak_ptr<IAsyncStream>>> _thread_map;
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

void ThreadRegistry::registerThread(ThreadType type, mo::IAsyncStream::Ptr_t stream)
{
    std::lock_guard<std::mutex> lock(_pimpl->mtx);
    auto& threads = _pimpl->_thread_map[type];
    auto pred = [stream](const std::weak_ptr<IAsyncStream> weak)
    {
        return weak.lock() == stream;
    };
    const size_t count = std::count_if(threads.begin(), threads.end(), pred);
    if (count == 0)
    {
        threads.push_back(stream);
    }
}

mo::IAsyncStream::Ptr_t ThreadRegistry::getThread(ThreadType type_)
{
    const int type = type_;
    std::lock_guard<std::mutex> lock(_pimpl->mtx);
    // TODO some kind of load balancing for multiple threads of a specific type
    IAsyncStream::Ptr_t current_thread = IAsyncStream::current();
    auto itr = _pimpl->_thread_map.find(type);
    if (itr != _pimpl->_thread_map.end())
    {
        if (!itr->second.empty())
        {
            auto pred = [current_thread](const std::weak_ptr<IAsyncStream>& weak)
            {
                return weak.lock() == current_thread;
            };
            if (std::count_if(itr->second.begin(), itr->second.end(), pred) == 0)
            {
                auto stream = itr->second.back().lock();
                if(stream)
                {
                    return stream;
                }
            }

            return current_thread;
        }
    }
    return nullptr;
}

ThreadRegistry* ThreadRegistry::instance()
{
    static ThreadRegistry* inst = nullptr;
    if (inst == nullptr)
    {
        inst = new ThreadRegistry();
    }
    return inst;
}
