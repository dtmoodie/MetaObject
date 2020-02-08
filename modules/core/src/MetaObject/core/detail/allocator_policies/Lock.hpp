#pragma once
#include <MetaObject/thread/Mutex.hpp>

namespace mo
{
    template <class BaseAllocator, class Mutex = Mutex_t>
    class LockPolicy : public BaseAllocator
    {
      public:
        uint8_t* allocate(size_t num_bytes, size_t elem_size);

        void deallocate(uint8_t* ptr, size_t num_bytes);

      private:
        Mutex m_mtx;
    };

    template <class BaseAllocator, class Mutex>
    uint8_t* LockPolicy<BaseAllocator, Mutex>::allocate(const size_t num_bytes, const size_t elem_size)
    {
        typename Mutex::Lock_t lock(m_mtx);
        return BaseAllocator::allocate(num_bytes, elem_size);
    }

    template <class BaseAllocator, class Mutex>
    void LockPolicy<BaseAllocator, Mutex>::deallocate(uint8_t* ptr, const size_t num_bytes)
    {
        typename Mutex::Lock_t lock(m_mtx);
        BaseAllocator::deallocate(ptr, num_bytes);
    }
} // namespace mo
