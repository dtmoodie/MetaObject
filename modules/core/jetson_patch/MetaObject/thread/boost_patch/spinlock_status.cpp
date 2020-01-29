#include "spinlock_status.hpp"

namespace std
{
  atomic<boost::fibers::detail::spinlock_status>::_Tp
  atomic<boost::fibers::detail::spinlock_status>::operator=(_Tp __i) noexcept
  { store(__i); return __i; }


  atomic<boost::fibers::detail::spinlock_status>::_Tp
  atomic<boost::fibers::detail::spinlock_status>::operator=(_Tp __i) volatile noexcept
  { store(__i); return __i; }


  bool
  atomic<boost::fibers::detail::spinlock_status>::is_lock_free() const noexcept
  { return __atomic_is_lock_free(sizeof(_M_i), nullptr); }


  bool
  atomic<boost::fibers::detail::spinlock_status>::is_lock_free() const volatile noexcept
  { return __atomic_is_lock_free(sizeof(_M_i), nullptr); }


  void
  atomic<boost::fibers::detail::spinlock_status>::store(_Tp __i, memory_order _m) noexcept
  { __atomic_store(&_M_i, &__i, _m); }


  void
  atomic<boost::fibers::detail::spinlock_status>::store(_Tp __i, memory_order _m) volatile noexcept
  { __atomic_store(&_M_i, &__i, _m); }


  atomic<boost::fibers::detail::spinlock_status>::_Tp
  atomic<boost::fibers::detail::spinlock_status>::load(memory_order _m) const noexcept
  {
    _Tp tmp;
    __atomic_load(const_cast<volatile int*>(&_M_i), &tmp, _m);
    return tmp;
  }


  atomic<boost::fibers::detail::spinlock_status>::_Tp
  atomic<boost::fibers::detail::spinlock_status>::load(memory_order _m) const volatile noexcept
  {
    _Tp tmp;
    __atomic_load(const_cast<volatile int*>(&_M_i), &tmp, _m);
    return tmp;
  }


  atomic<boost::fibers::detail::spinlock_status>::_Tp
  atomic<boost::fibers::detail::spinlock_status>::exchange(_Tp __i, memory_order _m) noexcept
  {
    _Tp tmp;
    __atomic_exchange(&_M_i, &__i, &tmp, _m);
    return tmp;
  }


  atomic<boost::fibers::detail::spinlock_status>::_Tp
  atomic<boost::fibers::detail::spinlock_status>::exchange(_Tp __i,
           memory_order _m) volatile noexcept
  {
    _Tp tmp;
    __atomic_exchange(&_M_i, &__i, &tmp, _m);
    return tmp;
  }


  bool
  atomic<boost::fibers::detail::spinlock_status>::compare_exchange_weak(_Tp& __e, _Tp __i, memory_order __s,
                        memory_order __f) noexcept
  {
    return __atomic_compare_exchange(&_M_i, &__e, &__i, true, __s, __f);
  }


  bool
  atomic<boost::fibers::detail::spinlock_status>::compare_exchange_weak(_Tp& __e, _Tp __i, memory_order __s,
                        memory_order __f) volatile noexcept
  {
    return __atomic_compare_exchange(&_M_i, &__e, &__i, true, __s, __f);
  }


  bool
  atomic<boost::fibers::detail::spinlock_status>::compare_exchange_weak(_Tp& __e, _Tp __i,
                        memory_order __m) noexcept
  { return compare_exchange_weak(__e, __i, __m, __m); }


  bool
  atomic<boost::fibers::detail::spinlock_status>::compare_exchange_weak(_Tp& __e, _Tp __i,
                 memory_order __m) volatile noexcept
  { return compare_exchange_weak(__e, __i, __m, __m); }


  bool
  atomic<boost::fibers::detail::spinlock_status>::compare_exchange_strong(_Tp& __e, _Tp __i, memory_order __s,
                          memory_order __f) noexcept
  {
    return __atomic_compare_exchange(&_M_i, &__e, &__i, false, __s, __f);
  }


  bool
  atomic<boost::fibers::detail::spinlock_status>::compare_exchange_strong(_Tp& __e, _Tp __i, memory_order __s,
                          memory_order __f) volatile noexcept
  {
    return __atomic_compare_exchange(&_M_i, &__e, &__i, false, __s, __f);
  }


  bool
  atomic<boost::fibers::detail::spinlock_status>::compare_exchange_strong(_Tp& __e, _Tp __i,
                           memory_order __m) noexcept
  { return compare_exchange_strong(__e, __i, __m, __m); }


  bool
  atomic<boost::fibers::detail::spinlock_status>::compare_exchange_strong(_Tp& __e, _Tp __i,
                 memory_order __m) volatile noexcept
  { return compare_exchange_strong(__e, __i, __m, __m); }
}
