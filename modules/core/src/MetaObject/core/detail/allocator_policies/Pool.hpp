#include "../MemoryBlock.hpp"
#include <list>
#include <memory>
namespace mo
{
    template <class XPU>
    class MemoryPool
    {
      public:
        static std::shared_ptr<MemoryPool> instance();
        static void setInstance(const std::shared_ptr<MemoryPool>& pool);
        MemoryPool();
        MemoryPool(const MemoryPool&) = delete;
        MemoryPool& operator=(const MemoryPool&) = delete;

        unsigned char* allocate(size_t num_bytes, size_t elem_size);

        void deallocate(unsigned char* ptr, size_t num_bytes);

        void release();

      private:
        size_t m_initial_block_size;
        std::list<std::unique_ptr<MemoryBlock<XPU>>> m_blocks;
    };

    template <class XPU>
    class PoolPolicy
    {
      public:
        PoolPolicy(const std::shared_ptr<MemoryPool<XPU>>& pool = MemoryPool<XPU>::instance());
        unsigned char* allocate(size_t num_bytes, size_t elem_size);

        void deallocate(unsigned char* ptr, size_t num_bytes);

        void release();

      private:
        std::shared_ptr<MemoryPool<XPU>> m_pool;
    };
    using CPUMemoryPool = MemoryPool<CPU>;
    using GPUMemoryPool = MemoryPool<GPU>;
}
