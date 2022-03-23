#include "Mutex.hpp"
#include <boost/fiber/mutex.hpp>
#include <boost/fiber/recursive_mutex.hpp>
#include <boost/fiber/recursive_timed_mutex.hpp>

namespace mo
{
    Mutex::Mutex()
    {
        m_mtx.reset(new boost::fibers::mutex());
    }

    Mutex::~Mutex()
    {
    }

    void Mutex::lock()
    {
        m_mtx->lock();
    }

    bool Mutex::try_lock()
    {
        return m_mtx->try_lock();
    }

    void Mutex::unlock()
    {
        m_mtx->unlock();
    }

    RecursiveMutex::RecursiveMutex()
    {
        m_mtx.reset(new boost::fibers::recursive_mutex());
    }

    RecursiveMutex::~RecursiveMutex() = default;

    void RecursiveMutex::lock()
    {
        m_mtx->lock();
    }

    bool RecursiveMutex::try_lock() noexcept
    {
        return m_mtx->try_lock();
    }

    void RecursiveMutex::unlock()
    {
        m_mtx->unlock();
    }

    RecursiveMutex::operator boost::fibers::recursive_mutex &()
    {
        return *m_mtx;
    }

    TimedMutex::TimedMutex()
    {
        m_mtx.reset(new boost::fibers::recursive_timed_mutex());
    }

    TimedMutex::~TimedMutex() = default;

    void TimedMutex::lock()
    {
        m_mtx->lock();
    }

    bool TimedMutex::try_lock() noexcept
    {
        return m_mtx->try_lock();
    }

    void TimedMutex::unlock()
    {
        m_mtx->unlock();
    }

    bool TimedMutex::try_lock_until_(std::chrono::steady_clock::time_point const& timeout_time) noexcept
    {
        return m_mtx->try_lock_until(timeout_time);
    }

    TimedMutex::operator boost::fibers::recursive_timed_mutex &()
    {
        return *m_mtx;
    }

} // namespace mo
