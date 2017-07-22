#include "MetaObject/thread/InterThread.hpp"
#include "MetaObject/core/detail/ConcurrentQueue.hpp"
#include "MetaObject/logging/logging.hpp"
#include <mutex>
#include <set>
#include <map>
using namespace mo;
struct impl {
#ifdef _DEBUG
    std::set<void*> _deleted_objects;

#endif
    struct Event {
        std::function<void(void)> function;
        void* obj;
    };
    struct QueueRegistery;
    struct EventQueue {
        void push(const Event& ev, size_t id) {
            queue.enqueue(ev);
            if(queue.size_approx() > 100)
                MO_LOG(warning) << "Event loop processing queue overflow " << id << " " << queue.size_approx();
            std::lock_guard<std::mutex> lock(mtx);
            if(callback)
                callback();
        }
        bool pop(Event& ev) {
            return queue.try_dequeue(ev);
        }
        void setCallback(const std::function<void(void)>& cb) {
            std::lock_guard<std::mutex> lock(mtx);
            callback = cb;
        }
        void remove(void* obj) {
            (void)obj;
        }
        size_t size() {
            return queue.size_approx();
        }

    private:
        friend struct QueueRegistery;
        moodycamel::ConcurrentQueue<Event> queue;
        std::function<void(void)> callback;
        std::mutex mtx;
    };

    struct QueueRegistery {
        static QueueRegistery& Instance() {
            static QueueRegistery* g_inst = nullptr;
            if(g_inst == nullptr)
                g_inst = new QueueRegistery();
            return *g_inst;
        }

        EventQueue* GetQueue(size_t thread) {
            std::unique_lock<std::mutex> lock;
            return &queues[thread];
        }
        void RemoveFromQueue(void* obj) {
            std::unique_lock<std::mutex> lock(mtx);
            for(auto& queue : queues) {
                queue.second.remove(obj);
            }
        }
        void clear() {
            std::unique_lock<std::mutex> lock(mtx);
            Event ev;
            for(auto& itr : queues) {
                while(itr.second.pop(ev)) {

                }
            }
        }

    private:
        std::map<size_t, EventQueue> queues;
        std::mutex mtx;
    };

    static impl* inst() {
        static impl* g_inst = nullptr;
        if(g_inst == nullptr)
            g_inst = new impl();
        return g_inst;
    }
    void register_notifier(const std::function<void(void)>& f, size_t id) {
        QueueRegistery::Instance().GetQueue(id)->setCallback(f);
    }
    void push(const std::function<void(void)>& f, size_t id, void* obj) {
        if(getThisThread() == id) {
            f();
            return;
        }
        QueueRegistery::Instance().GetQueue(id)->push({f, obj}, id);
    }
    int run(size_t id) {
        auto queue = QueueRegistery::Instance().GetQueue(id);
        Event ev;
        int count = 0;
        while(queue->pop(ev)) {
            ev.function();
            ++count;
        }
        return count;
    }
    bool run_once(size_t id) {
        auto queue = QueueRegistery::Instance().GetQueue(id);
        Event ev;
        if(queue->pop(ev)) {
            ev.function();
            return true;
        }
        return false;
    }
    void remove_from_queue(void* obj) {
        QueueRegistery::Instance().RemoveFromQueue(obj);
    }
    size_t size(size_t id) {
        return QueueRegistery::Instance().GetQueue(id)->size();
    }
    void Cleanup() {
        QueueRegistery::Instance().clear();
    }
};
void ThreadSpecificQueue::push(const std::function<void(void)>& f, size_t id, void* obj) {
    impl::inst()->push(f, id, obj);
}
int ThreadSpecificQueue::run(size_t id) {
    return impl::inst()->run(id);
}
void ThreadSpecificQueue::registerNotifier(const std::function<void(void)>& f, size_t id) {
    impl::inst()->register_notifier(f, id);
}
bool ThreadSpecificQueue::runOnce(size_t id) {
    return impl::inst()->run_once(id);
}
void ThreadSpecificQueue::removeFromQueue(void* obj) {
    impl::inst()->remove_from_queue(obj);
}
size_t ThreadSpecificQueue::size(size_t id) {
    return impl::inst()->size(id);
}
void ThreadSpecificQueue::cleanup() {
    impl::inst()->Cleanup();
}
