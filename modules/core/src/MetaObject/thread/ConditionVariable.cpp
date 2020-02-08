#include "ConditionVariable.hpp"

#include <boost/fiber/condition_variable.hpp>

namespace mo
{
    ConditionVariable::ConditionVariable()
    {
        m_cv.reset(new boost::fibers::condition_variable_any());
    }

    ConditionVariable::~ConditionVariable() = default;

    void ConditionVariable::notify_one() noexcept
    {
        m_cv->notify_one();
    }

    void ConditionVariable::notify_all() noexcept
    {
        m_cv->notify_all();
    }

    void ConditionVariable::wait(Mutex::Lock_t& lt)
    {
        m_cv->wait(lt);
    }

    void ConditionVariable::wait(RecursiveMutex::Lock_t& lt)
    {
        m_cv->wait(lt);
    }

    void ConditionVariable::wait(TimedMutex::Lock_t& lt)
    {
        m_cv->wait(lt);
    }

    ConditionVariable::STATUS ConditionVariable::wait_until(RecursiveMutex::Lock_t& lt,
                                                            const std::chrono::steady_clock::time_point& timeout_time_)
    {
        m_cv->wait_until(lt, timeout_time_);
    }

    ConditionVariable::STATUS ConditionVariable::wait_until(TimedMutex::Lock_t& lt,
                                                            const std::chrono::steady_clock::time_point& timeout_time_)
    {
        m_cv->wait_until(lt, timeout_time_);
    }

    ConditionVariable::STATUS ConditionVariable::wait_until(Mutex::Lock_t& lt,
                                                            const std::chrono::steady_clock::time_point& timeout_time_)
    {
        m_cv->wait_until(lt, timeout_time_);
    }

    ConditionVariable::STATUS ConditionVariable::wait_for(RecursiveMutex::Lock_t& lt,
                                                          const Duration_t& timeout_duration)
    {
        m_cv->wait_for(lt, timeout_duration);
    }

    ConditionVariable::STATUS ConditionVariable::wait_for(TimedMutex::Lock_t& lt, const Duration_t& timeout_duration)
    {
        m_cv->wait_for(lt, timeout_duration);
    }

    ConditionVariable::STATUS ConditionVariable::wait_for(Mutex::Lock_t& lt, const Duration_t& timeout_duration)
    {
        m_cv->wait_for(lt, timeout_duration);
    }
} // namespace mo
