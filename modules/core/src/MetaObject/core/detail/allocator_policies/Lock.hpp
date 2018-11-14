#pragma once
#include <boost/fiber/mutex.hpp>
namespace mo
{
    template <class Allocator, class Mutex = boost::fibers::mutex>
    class LockPolicy : public Allocator
    {
      public:
        uint8_t* allocate(const uint64_t num_bytes, const uint64_t elem_size);

        void deallocate(uint8_t* ptr, const uint64_t num_bytes);

      private:
        Mutex m_mtx;
    };

    template <class Allocator, class Mutex>
    uint8_t* LockPolicy<Allocator, Mutex>::allocate(const uint64_t num_bytes, const uint64_t elem_size)
    {
        typename Mutex::scoped_lock lock(m_mtx);
        return Allocator::allocate(num_bytes, elem_size);
    }

    template <class Allocator, class Mutex>
    void LockPolicy<Allocator, Mutex>::deallocate(uint8_t* ptr, const uint64_t num_bytes)
    {
        typename Mutex::scoped_lock lock(m_mtx);
        Allocator::deallocate(ptr, num_bytes);
    }
}
