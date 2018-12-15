
//          Copyright Oliver Kowalke 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_FIBERS_SPINLOCK_STATUS_H
#define BOOST_FIBERS_SPINLOCK_STATUS_H

namespace boost {
namespace fibers {
namespace detail {

enum  spinlock_status: int {
    locked = 0,
    unlocked
};

}}}

#include <atomic>

namespace std
{
  template<>
  struct atomic<boost::fibers::detail::spinlock_status>
  {
  private:
    using _Tp = int;
    volatile _Tp _M_i;

  public:
    atomic() noexcept = default;
    ~atomic() noexcept = default;
    atomic(const atomic&) = delete;
    atomic& operator=(const atomic&) = delete;
    atomic& operator=(const atomic&) volatile = delete;


    constexpr atomic(_Tp __i) noexcept : _M_i(__i) { }

    operator _Tp() const noexcept
    { return load(); }

    operator _Tp() const volatile noexcept
    { return load(); }

    _Tp
    operator=(_Tp __i) noexcept;

    _Tp
    operator=(_Tp __i) volatile noexcept;

    bool
    is_lock_free() const noexcept;

    bool
    is_lock_free() const volatile noexcept;

    void
    store(_Tp __i, memory_order _m = memory_order_seq_cst) noexcept;

    void
    store(_Tp __i, memory_order _m = memory_order_seq_cst) volatile noexcept;

    _Tp
    load(memory_order _m = memory_order_seq_cst) const noexcept;

    _Tp
    load(memory_order _m = memory_order_seq_cst) const volatile noexcept;

    _Tp
    exchange(_Tp __i, memory_order _m = memory_order_seq_cst) noexcept;

    _Tp
    exchange(_Tp __i,
             memory_order _m = memory_order_seq_cst) volatile noexcept;

    bool
    compare_exchange_weak(_Tp& __e, _Tp __i, memory_order __s,
                          memory_order __f) noexcept;

    bool
    compare_exchange_weak(_Tp& __e, _Tp __i, memory_order __s,
                          memory_order __f) volatile noexcept;

    bool
    compare_exchange_weak(_Tp& __e, _Tp __i,
                          memory_order __m = memory_order_seq_cst) noexcept;

    bool
    compare_exchange_weak(_Tp& __e, _Tp __i,
                   memory_order __m = memory_order_seq_cst) volatile noexcept;

    bool
    compare_exchange_strong(_Tp& __e, _Tp __i, memory_order __s,
                            memory_order __f) noexcept;

    bool
    compare_exchange_strong(_Tp& __e, _Tp __i, memory_order __s,
                            memory_order __f) volatile noexcept;

    bool
    compare_exchange_strong(_Tp& __e, _Tp __i,
                             memory_order __m = memory_order_seq_cst) noexcept;

    bool
    compare_exchange_strong(_Tp& __e, _Tp __i,
                   memory_order __m = memory_order_seq_cst) volatile noexcept;
  };


}

#endif // BOOST_FIBERS_SPINLOCK_STATUS_H
