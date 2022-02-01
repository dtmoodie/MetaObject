#include "MetaObject/thread/ThreadPool.hpp"
#include "MetaObject/thread/FiberScheduler.hpp"
#include "MetaObject/thread/Thread.hpp"

using namespace mo;

ThreadPool::~ThreadPool()
{
    /*for (auto sched : m_schedulers)
    {
        sched->releaseAssistant();
    }*/
    cleanup();
    m_running_threads.clear();
}

std::shared_ptr<Thread> ThreadPool::requestThread()
{
    std::lock_guard<std::mutex> lock(m_mtx);
    if (m_free_threads.empty())
    {
        std::shared_ptr<Thread> owning_ptr(new Thread(this));
        m_running_threads.push_back(owning_ptr);
        return std::shared_ptr<Thread>(owning_ptr.get(), [this, owning_ptr](Thread*) { returnThread(owning_ptr); });
    }
    auto owning_ptr = m_free_threads.front();
    m_free_threads.pop_front();
    m_running_threads.push_back(owning_ptr);
    return std::shared_ptr<Thread>(owning_ptr.get(), [this, owning_ptr](Thread*) { returnThread(owning_ptr); });
}

void ThreadPool::cleanup()
{
    m_free_threads.clear();
}

void ThreadPool::returnThread(const std::shared_ptr<Thread>& thread)
{
    std::lock_guard<std::mutex> lock(m_mtx);
    m_free_threads.push_back(thread);
    auto itr = std::find(m_running_threads.begin(), m_running_threads.end(), thread);
    if (itr != m_running_threads.end())
    {
        m_running_threads.erase(itr);
    }
}

void ThreadPool::addScheduler(PriorityScheduler* sched)
{
    std::lock_guard<std::mutex> lock(m_mtx);
    m_schedulers.insert(sched);
}

void ThreadPool::removeScheduler(PriorityScheduler* sched)
{
    std::lock_guard<std::mutex> lock(m_mtx);
    auto itr = m_schedulers.find(sched);
    if (itr != m_schedulers.end())
    {
        m_schedulers.erase(itr);
    }
}

std::vector<PriorityScheduler*> ThreadPool::getSchedulers() const
{
    std::lock_guard<std::mutex> lock(m_mtx);
    return {m_schedulers.begin(), m_schedulers.end()};
}
