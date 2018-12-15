#pragma once
#include <MetaObject/thread/fiber_include.hpp>

namespace mo
{
    template <class BaseAllocator, class Mutex = boost::fibers::mutex>
    class LockPolicy : public BaseAllocator
    {
      public:
        uint8_t* allocate(const uint64_t num_bytes, const uint64_t elem_size);

        void deallocate(uint8_t* ptr, const uint64_t num_bytes);

      private:
        Mutex m_mtx;
    };

    template <class BaseAllocator, class Mutex>
    uint8_t* LockPolicy<BaseAllocator, Mutex>::allocate(const uint64_t num_bytes, const uint64_t elem_size)
    {
        std::lock_guard<Mutex> lock(m_mtx);
        return BaseAllocator::allocate(num_bytes, elem_size);
    }

    template <class BaseAllocator, class Mutex>
    void LockPolicy<BaseAllocator, Mutex>::deallocate(uint8_t* ptr, const uint64_t num_bytes)
    {
        std::lock_guard<Mutex> lock(m_mtx);
        BaseAllocator::deallocate(ptr, num_bytes);
    }
}
