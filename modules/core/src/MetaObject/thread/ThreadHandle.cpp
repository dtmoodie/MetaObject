#include "MetaObject/thread/ThreadHandle.hpp"
#include "MetaObject/logging/profiling.hpp"
#include "MetaObject/thread/Thread.hpp"
#include "MetaObject/thread/ThreadPool.hpp"

using namespace mo;
ThreadHandle::ThreadHandle(const std::shared_ptr<Thread>& thread)
    : m_thread(thread)
{
}

ContextPtr_t ThreadHandle::context() const
{
    if (m_thread)
    {
        return m_thread->context();
    }
    return {};
}

bool ThreadHandle::pushWork(std::function<void(void)>&& f)
{
    if (m_thread)
    {

        return true;
    }
    return false;
}

void ThreadHandle::setExitCallback(std::function<void(void)>&& f)
{
    if (m_thread)
    {
        m_thread->setExitCallback(std::move(f));
    }
}

size_t ThreadHandle::threadId() const
{
    if (m_thread)
    {
        return m_thread->threadId();
    }
    return 0;
}

bool ThreadHandle::isOnThread() const
{
    if (m_thread)
    {
        return m_thread->isOnThread();
    }
    return false;
}

void ThreadHandle::setThreadName(const std::string& name)
{
    if (m_thread)
    {
        m_thread->setName(name);
        if (m_thread->isOnThread())
        {
            m_thread->setName(name);
        }
        else
        {
            auto thread = m_thread;
            std::string name_ = name;
            // TODO fibers
            // m_thread->pushEventQueue(EventToken([name_, thread, this]() { m_thread->setName(name_); }));
        }
    }
}

const std::string& ThreadHandle::threadName() const
{
    thread_local std::string default_name("No thread");
    if (m_thread)
    {
        return m_thread->threadName();
    }
    return default_name;
}
