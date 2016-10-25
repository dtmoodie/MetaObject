#include "MetaObject/Thread/InterThread.hpp"
#include "MetaObject/Detail/ConcurrentQueue.hpp"
#include "MetaObject/Logging/Log.hpp"
#include <mutex>
#include <set>
#include <map>
using namespace mo;
struct impl
{
#ifdef _DEBUG
    std::set<void*> _deleted_objects;

#endif
    std::map<size_t,  // Thread id
        std::tuple<
            ConcurrentQueue<std::pair<std::function<void(void)>, void*>>,  // queue of function + obj*
            std::function<void(void)>, std::mutex> // callback on push to queue
    > thread_queues;

    std::mutex mtx;

    static impl* inst()
    {
        static impl g_inst;
        return &g_inst;        
    }
    void register_notifier(const std::function<void(void)>& f, size_t id)
    {
        std::lock_guard<std::mutex> lock(mtx);
        std::get<1>(thread_queues[id]) = f;
    }
    void push(const std::function<void(void)>& f, size_t id, void* obj)
    {
        if(GetThisThread() == id)
        {
            f();
            return;
        }
        std::tuple<ConcurrentQueue<std::pair<std::function<void(void)>, void*>>, std::function<void(void)>, std::mutex>* queue = nullptr;
        {
            std::lock_guard<std::mutex> lock(mtx);
            queue = &thread_queues[id];
        }
        std::lock_guard<std::mutex> lock(std::get<2>(*queue));
        if(std::get<0>(*queue).size() > 100)
            LOG(warning) << "Event loop processing queue overflow " << std::get<0>(*queue).size() << " for thread " << id;
        std::get<0>(*queue).push(std::pair<std::function<void(void)>, void*>(f, obj));
        if (std::get<1>(*queue))
            std::get<1>(*queue)();
    }
    void run(size_t id)
    {
        std::tuple<ConcurrentQueue<std::pair<std::function<void(void)>, void*>>, std::function<void(void)>, std::mutex>* queue = nullptr;
        {
            std::lock_guard<std::mutex> lock(mtx);
            queue = &thread_queues[id];
        }
        std::lock_guard<std::mutex> lock(std::get<2>(*queue));
        std::pair<std::function<void(void)>, void*> f;
        while (std::get<0>(*queue).try_pop(f))
        {
            f.first();
        }
    }
    void run_once(size_t id)
    {
        std::tuple<ConcurrentQueue<std::pair<std::function<void(void)>, void*>>, std::function<void(void)>, std::mutex>* queue = nullptr;
        {
            std::lock_guard<std::mutex> lock(mtx);
            queue = &thread_queues[id];
        }
        std::pair<std::function<void(void)>, void*> f;
        std::lock_guard<std::mutex> lock(std::get<2>(*queue));
        if(std::get<0>(*queue).try_pop(f))
        {
            f.first();
        }
    }
    void remove_from_queue(void* obj)
    {
        std::lock_guard<std::mutex> lock(mtx);
        for (auto& queue: thread_queues)
        {
            std::lock_guard<std::mutex> lock(std::get<2>(queue.second));
            for (auto itr = std::get<0>(queue.second).begin(); itr != std::get<0>(queue.second).end(); )
            {
                if(itr->second == obj)
                {
                    LOG(trace) << "Removing item from queue for object: " << obj;
                    itr = std::get<0>(queue.second).erase(itr);
                }
                else
                {
                    ++itr;
                }
            }
        }
    }
};
void ThreadSpecificQueue::Push(const std::function<void(void)>& f, size_t id, void* obj)
{
    
#ifdef _DEBUG
    if(impl::inst()->_deleted_objects.find(obj) != impl::inst()->_deleted_objects.end())
    {
        LOG(trace) << "Pushing function onto queue from deleted object";
        //return;
    }
#endif
    impl::inst()->push(f, id, obj);
}
void ThreadSpecificQueue::Run(size_t id)
{
    impl::inst()->run(id);
}
void ThreadSpecificQueue::RegisterNotifier(const std::function<void(void)>& f, size_t id)
{
    impl::inst()->register_notifier(f, id);
}
void ThreadSpecificQueue::RunOnce(size_t id)
{
    impl::inst()->run_once(id);
}
void ThreadSpecificQueue::RemoveFromQueue(void* obj)
{
    impl::inst()->remove_from_queue(obj);
#ifdef _DEBUG
    impl::inst()->_deleted_objects.insert(obj);
#endif
}