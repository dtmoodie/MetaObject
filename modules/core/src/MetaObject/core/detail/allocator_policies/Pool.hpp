#pragma once
#include "../Allocator.hpp"
#include "../MemoryBlock.hpp"
#include <list>
#include <memory>
namespace mo
{
    // Memory pool is not copyable due to the unique ptr to memory blocks
    template <class XPU>
    class MemoryPool : public XPU::Allocator_t
    {
      public:
        using Allocator_t = typename XPU::Allocator_t;
        using Ptr_t = std::shared_ptr<MemoryPool<XPU>>;

        static Ptr_t create();

        uint8_t* allocate(size_t num_bytes, size_t elem_size);
        void deallocate(uint8_t* ptr, size_t num_bytes);
        void release();

      private:
        size_t m_initial_block_size = 20 * 1024 * 1024;
        std::list<std::unique_ptr<MemoryBlock<XPU>>> m_blocks;
    };

    // PoolPolicy is copyable since it holds a shared pointer to the pool
    template <class XPU>
    class PoolPolicy : public XPU::Allocator_t
    {
      public:
        using Allocator_t = typename XPU::Allocator_t;
        PoolPolicy(const std::shared_ptr<MemoryPool<XPU>>& pool = MemoryPool<XPU>::create());

        uint8_t* allocate(size_t num_bytes, size_t elem_size) override;

        void deallocate(uint8_t* ptr, size_t num_bytes) override;

        void release() override;

      private:
        std::shared_ptr<MemoryPool<XPU>> m_pool;
    };

    using CPUMemoryPool = MemoryPool<CPU>;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    /// Implementation
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template <class XPU>
    typename MemoryPool<XPU>::Ptr_t MemoryPool<XPU>::create()
    {
        return std::make_shared<MemoryPool<XPU>>();
    }

    template <class XPU>
    uint8_t* MemoryPool<XPU>::allocate(const size_t num_bytes, const size_t elem_size)
    {
        uint8_t* ptr = nullptr;
        for (auto& itr : m_blocks)
        {
            ptr = itr->allocate(num_bytes, elem_size);
            if (ptr)
            {
                return ptr;
            }
        }
        // If we get to this point, then no memory was found, need to allocate new memory
        m_blocks.push_back(std::unique_ptr<MemoryBlock<XPU>>(
            new MemoryBlock<XPU>(std::max(m_initial_block_size / 2, num_bytes + elem_size))));
        ptr = (*m_blocks.rbegin())->allocate(num_bytes, elem_size);
        if (ptr)
        {
            return ptr;
        }
        // Should only ever reach this if we can't allocate more memory for some reason
        return nullptr;
    }

    template <class XPU>
    void MemoryPool<XPU>::deallocate(uint8_t* ptr, const size_t num_bytes)
    {
        for (auto& itr : m_blocks)
        {
            if (itr->deallocate(ptr, num_bytes))
            {
                return;
            }
        }
    }

    template <class XPU>
    void MemoryPool<XPU>::release()
    {
        m_blocks.clear();
    }

    template <class XPU>
    PoolPolicy<XPU>::PoolPolicy(const std::shared_ptr<MemoryPool<XPU>>& pool)
        : m_pool(pool)
    {
    }

    template <class XPU>
    uint8_t* PoolPolicy<XPU>::allocate(const size_t num_bytes, const size_t elem_size)
    {
        return m_pool->allocate(num_bytes, elem_size);
    }

    template <class XPU>
    void PoolPolicy<XPU>::deallocate(uint8_t* ptr, const size_t num_bytes)
    {
        m_pool->deallocate(ptr, num_bytes);
    }

    template <class XPU>
    void PoolPolicy<XPU>::release()
    {
        m_pool->release();
    }
}
