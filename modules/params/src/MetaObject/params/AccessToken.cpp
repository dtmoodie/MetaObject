#include "MetaObject/params/AccessToken.hpp"
#include <MetaObject/thread/Mutex.hpp>
#include <MetaObject/thread/fiber_include.hpp>
#include <boost/thread/locks.hpp>

namespace mo
{
    AccessTokenLock::AccessTokenLock()
    {
    }

    AccessTokenLock::AccessTokenLock(AccessTokenLock&& other)
        : lock(std::move(other.lock))
    {
    }

    AccessTokenLock::AccessTokenLock(Lock_t&& lock)
        : lock(new Lock_t(std::move(lock)))
    {
    }

    AccessTokenLock::AccessTokenLock(Mutex_t& mtx)
        : lock(new Lock_t(mtx))
    {
    }

    AccessTokenLock::~AccessTokenLock()
    {
    }
} // namespace mo
