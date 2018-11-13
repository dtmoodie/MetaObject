#pragma once
#include <MetaObject/core/Event.hpp>
#include <MetaObject/core/detail/Forward.hpp>
#include <MetaObject/detail/Export.hpp>

#include <functional>
#include <memory>
namespace mo
{
    class Thread;
    class ThreadPool;
    class Context;
    class ISlot;
    class Connection;

    // Add a level of indirection such that boost/thread header files do not need to be viewed by nvcc
    // this is a work around for older versions of nvcc that would implode on including certain boost headers
    class MO_EXPORTS ThreadHandle
    {
      public:
        ThreadHandle(const std::shared_ptr<Thread>& thread);

        ContextPtr_t context() const;
        size_t threadId() const;

        const std::string& threadName() const;
        bool isOnThread() const;

        // Events must be handled on the enqueued thread
        bool pushEventQueue(EventToken&& token);
        bool pushEventQueue(std::function<void(void)>&& f, const uint64_t id = 0);
        template <class T, class... Args>
        bool pushEventQueue(T* obj, void (T::*fptr)(Args...), Args&&... args)
        {
            return pushEventQueue(EventToken::createCallback(obj, fptr, std::forward<Args>(args)...));
        }
        // Work can be stolen and can exist on any thread
        bool pushWork(std::function<void(void)>&& f);

        void setExitCallback(std::function<void(void)>&& f);
        void setThreadName(const std::string& name);

      private:
        std::shared_ptr<Thread> m_thread;
    };
}
