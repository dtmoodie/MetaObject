#include "MetaObject/params/AccessToken.hpp"
#include <boost/thread/lock_guard.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/recursive_mutex.hpp>

namespace mo
{
    AccessTokenLock::AccessTokenLock() {}
    AccessTokenLock::AccessTokenLock(AccessTokenLock&& other) : lock(std::move(other.lock)) {}
    AccessTokenLock::AccessTokenLock(Mutex_t& mtx) { lock = std::unique_ptr<boost::lock_guard<mo::Mutex_t>>(new boost::lock_guard<mo::Mutex_t>(mtx)); }

    AccessTokenLock::~AccessTokenLock() {}
}
