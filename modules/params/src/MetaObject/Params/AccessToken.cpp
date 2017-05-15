#include "MetaObject/params/AccessToken.hpp"
#include <boost/thread/mutex.hpp>
#include <boost/thread/lock_guard.hpp>
#include <boost/thread/recursive_mutex.hpp>

namespace mo{
    AccessTokenLock::AccessTokenLock():
        lock(nullptr){
    }
    AccessTokenLock::AccessTokenLock(AccessTokenLock& other):
        lock(other.lock){
        other.lock = nullptr;
    }
    AccessTokenLock::AccessTokenLock(AccessTokenLock&& other):
    lock(other.lock){
        other.lock = nullptr;
    }
    AccessTokenLock::AccessTokenLock(Mutex_t& mtx){
        lock = new boost::lock_guard<mo::Mutex_t>(mtx);
    }

    AccessTokenLock::~AccessTokenLock(){
        if(lock)
            delete lock;
    }
}