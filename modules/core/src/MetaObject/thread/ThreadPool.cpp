#include "MetaObject/thread/ThreadPool.hpp"
#include "MetaObject/thread/Thread.hpp"
using namespace mo;

ThreadPool::~ThreadPool()
{
    cleanup();
}

std::shared_ptr<Thread> ThreadPool::requestThread()
{
    std::lock_guard<std::mutex> lock(m_mtx);
    if (m_threads.empty())
    {
        std::shared_ptr<Thread> owning_ptr(new Thread(this));
        return std::shared_ptr<Thread>(owning_ptr.get(), [this, owning_ptr](Thread*) { returnThread(owning_ptr); });
    }
    else
    {
        auto owning_ptr = m_threads.front();
        m_threads.pop_front();
        return std::shared_ptr<Thread>(owning_ptr.get(), [this, owning_ptr](Thread*) { returnThread(owning_ptr); });
    }
}

void ThreadPool::cleanup()
{
    m_threads.clear();
}

void ThreadPool::returnThread(const std::shared_ptr<Thread>& thread)
{
    std::lock_guard<std::mutex> lock(m_mtx);
    m_threads.push_back(thread);
}

void ThreadPool::addScheduler(PriorityScheduler* sched)
{
    std::lock_guard<std::mutex> lock(m_mtx);
    m_schedulers.push_back(sched);
}

void ThreadPool::removeScheduler(PriorityScheduler* sched)
{
    std::lock_guard<std::mutex> lock(m_mtx);
    std::remove(m_schedulers.begin(), m_schedulers.end(), sched);
}

std::vector<PriorityScheduler*> ThreadPool::getSchedulers() const
{
    std::lock_guard<std::mutex> lock(m_mtx);
    return m_schedulers;
}
