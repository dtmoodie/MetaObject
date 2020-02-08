#ifndef MO_THREAD_CONDITION_VARIABLE_HPP
#define MO_THREAD_CONDITION_VARIABLE_HPP
#include "Mutex.hpp"

#include <functional>
#include <memory>

namespace boost
{
    namespace fibers
    {
        class condition_variable_any;
    } // namespace fibers
} // namespace boost

namespace mo
{
    struct Mutex;
    struct RecursiveMutex;
    struct TimedMutex;

    struct MO_EXPORTS ConditionVariable
    {
        using Duration_t = std::chrono::duration<std::chrono::nanoseconds::rep, std::chrono::nanoseconds::period>;
        enum class STATUS
        {
            kNO_TIMEOUT = 1,
            kTIMEOUT
        };

        ConditionVariable();
        ~ConditionVariable();

        void notify_one() noexcept;

        void notify_all() noexcept;

        void wait(RecursiveMutex::Lock_t& lt);
        void wait(Mutex::Lock_t& lt);
        void wait(TimedMutex::Lock_t& lt);

        template <class PREDICATE>
        void wait(RecursiveMutex::Lock_t& lt, PREDICATE pred);
        template <class PREDICATE>
        void wait(TimedMutex::Lock_t& lt, PREDICATE pred);

        STATUS wait_until(Mutex::Lock_t& lt, const std::chrono::steady_clock::time_point& timeout_time_);
        STATUS wait_until(RecursiveMutex::Lock_t& lt, const std::chrono::steady_clock::time_point& timeout_time_);
        STATUS wait_until(TimedMutex::Lock_t& lt, const std::chrono::steady_clock::time_point& timeout_time_);

        STATUS wait_for(Mutex::Lock_t& lt, const Duration_t& timeout_duration);
        STATUS wait_for(RecursiveMutex::Lock_t& lt, const Duration_t& timeout_duration);
        STATUS wait_for(TimedMutex::Lock_t& lt, const Duration_t& timeout_duration);

      private:
        std::unique_ptr<boost::fibers::condition_variable_any> m_cv;
    };
} // namespace mo

#endif // MO_THREAD_CONDITION_VARIABLE_HPP