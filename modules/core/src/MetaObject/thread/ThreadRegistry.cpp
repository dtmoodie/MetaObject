#include "ThreadRegistry.hpp"
#include "ThreadPool.hpp"

#include "MetaObject/core/SystemTable.hpp"

#include <algorithm>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

namespace mo
{
    struct ThreadRegistryImpl : public ThreadRegistry
    {
        ThreadRegistryImpl()
        {
            // finish any work
            for (auto& thread : m_worker_threads)
            {
                IAsyncStreamPtr_t stream = thread->asyncStream();
                stream->synchronize();
            }
        }

        IAsyncStream::Ptr_t getGUIStream() override
        {
            IAsyncStream::Ptr_t out;

            {
                std::lock_guard<std::mutex> lock(mtx);
                out = this->m_gui_stream.lock();
                if (!out)
                {
                    out = mo::IAsyncStream::create();
                    m_gui_stream = out;
                }
            }
            return out;
        }

        void setGUIStream(IAsyncStream::Ptr_t stream) override
        {
            std::lock_guard<std::mutex> lock(mtx);
            m_gui_stream = stream;
        }

        void registerThread(ThreadType type, IAsyncStream::Ptr_t stream) override
        {
            std::lock_guard<std::mutex> lock(mtx);
            auto& threads = _thread_map[type];
            auto pred = [stream](const std::weak_ptr<IAsyncStream> weak) { return weak.lock() == stream; };
            const size_t count = std::count_if(threads.begin(), threads.end(), pred);
            if (count == 0)
            {
                threads.push_back(stream);
            }
        }

        IAsyncStream::Ptr_t getThread(ThreadType type_) override
        {
            const int type = type_;
            std::lock_guard<std::mutex> lock(mtx);
            // TODO some kind of load balancing for multiple threads of a specific type
            IAsyncStream::Ptr_t current_thread = IAsyncStream::current();
            auto itr = _thread_map.find(type);
            if (itr != _thread_map.end())
            {
                if (!itr->second.empty())
                {
                    auto pred = [current_thread](const std::weak_ptr<IAsyncStream>& weak) {
                        return weak.lock() == current_thread;
                    };
                    if (std::count_if(itr->second.begin(), itr->second.end(), pred) == 0)
                    {
                        auto stream = itr->second.back().lock();
                        if (stream)
                        {
                            return stream;
                        }
                    }

                    return current_thread;
                }
            }
            if (type_ == WORKER)
            {
                // need to create a thread
                std::shared_ptr<ThreadPool> pool = SystemTable::instance()->getSingleton<mo::ThreadPool>();
                std::shared_ptr<mo::Thread> thread = std::make_shared<mo::Thread>(pool.get());
                IAsyncStreamPtr_t output = thread->asyncStream();
                _thread_map[type_].push_back(output);
                m_worker_threads.push_back(thread);
                return output;
            }
            return nullptr;
        }

        std::map<int, std::vector<std::weak_ptr<IAsyncStream>>> _thread_map;
        std::weak_ptr<IAsyncStream> m_gui_stream;
        std::vector<std::shared_ptr<Thread>> m_worker_threads;
        mutable std::mutex mtx;
    };

    size_t tid()
    {
        std::stringstream ss;
        ss << std::this_thread::get_id();
        size_t output;
        ss >> output;
        return output;
    }

    size_t getThisThread()
    {
        thread_local size_t thread_id = tid();
        return thread_id;
    }

    ThreadRegistry::~ThreadRegistry()
    {
    }

    ThreadRegistry* ThreadRegistry::instance()
    {
        return instance(SystemTable::instance().get());
    }

    ThreadRegistry* ThreadRegistry::instance(SystemTable* table)
    {
        return table->getSingleton<ThreadRegistry, ThreadRegistryImpl>().get();
    }

} // namespace mo
