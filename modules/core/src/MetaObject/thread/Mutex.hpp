#ifndef MO_CORE_MUTEX_HPP
#define MO_CORE_MUTEX_HPP
#include <MetaObject/core.hpp>

#include <chrono>
#include <memory>
#include <mutex>

namespace boost
{
    namespace fibers
    {
        class recursive_timed_mutex;
        class recursive_mutex;
        class mutex;
    } // namespace fibers
} // namespace boost

namespace mo
{
    struct MO_EXPORTS Mutex
    {
        using Lock_t = std::unique_lock<Mutex>;
        Mutex();
        ~Mutex();

        void lock();

        bool try_lock();

        void unlock();

      private:
        std::unique_ptr<boost::fibers::mutex> m_mtx;
    };

    struct MO_EXPORTS RecursiveMutex
    {
        using Lock_t = std::unique_lock<RecursiveMutex>;
        RecursiveMutex();
        ~RecursiveMutex();
        void lock();

        bool try_lock() noexcept;

        void unlock();

        operator boost::fibers::recursive_mutex &();

      private:
        std::unique_ptr<boost::fibers::recursive_mutex> m_mtx;
    };

    struct MO_EXPORTS TimedMutex
    {
        using Lock_t = std::unique_lock<TimedMutex>;
        TimedMutex();
        ~TimedMutex();

        void lock();

        bool try_lock() noexcept;

        template <typename Clock, typename Duration>
        bool try_lock_until(std::chrono::time_point<Clock, Duration> const& timeout_time_);

        template <typename Rep, typename Period>
        bool try_lock_for(std::chrono::duration<Rep, Period> const& timeout_duration);

        void unlock();

        operator boost::fibers::recursive_timed_mutex &();

      private:
        bool try_lock_until_(std::chrono::steady_clock::time_point const& timeout_time) noexcept;
        std::unique_ptr<boost::fibers::recursive_timed_mutex> m_mtx;
    };

    template <typename Clock, typename Duration>
    bool TimedMutex::try_lock_until(std::chrono::time_point<Clock, Duration> const& timeout_time_)
    {
        std::chrono::steady_clock::time_point timeout_time =
            std::chrono::steady_clock::now() + (timeout_time_ - Clock::now());
        return try_lock_until_(timeout_time);
    }

    template <typename Rep, typename Period>
    bool TimedMutex::try_lock_for(std::chrono::duration<Rep, Period> const& timeout_duration)
    {
        return try_lock_until_(std::chrono::steady_clock::now() + timeout_duration);
    }
} // namespace mo

#endif // MO_CORE_MUTEX_HPP