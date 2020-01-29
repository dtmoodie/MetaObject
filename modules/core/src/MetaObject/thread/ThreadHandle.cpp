#include "ThreadHandle.hpp"
#include <MetaObject/thread/FiberProperties.hpp>
#include <MetaObject/thread/Thread.hpp>

#include <MetaObject/thread/fiber_include.hpp>

namespace mo
{
    ThreadHandle::ThreadHandle(const std::shared_ptr<Thread>& thread)
        : m_thread(thread)
    {
    }

    IAsyncStreamPtr_t ThreadHandle::asyncStream(const Duration timeout) const
    {
        return m_thread->asyncStream(timeout);
    }

    size_t ThreadHandle::threadId() const
    {
        return m_thread->threadId();
    }

    bool ThreadHandle::isOnThread() const
    {
        return m_thread->isOnThread();
    }

    const std::string& ThreadHandle::threadName() const
    {
        return m_thread->threadName();
    }

    void ThreadHandle::setName(const std::string& name)
    {
        m_thread->setName(name);
    }
}
