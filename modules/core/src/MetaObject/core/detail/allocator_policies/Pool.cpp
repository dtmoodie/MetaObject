#include "Pool-inl.hpp"

namespace mo
{
    std::weak_ptr<MemoryPool<CPU>> g_cpu_memory_pool;
    std::weak_ptr<MemoryPool<GPU>> g_gpu_memory_pool;

    template <>
    std::shared_ptr<MemoryPool<CPU>> MemoryPool<CPU>::instance()
    {
        auto out = g_cpu_memory_pool.lock();
        if (!out)
        {
            out = std::make_shared<MemoryPool<CPU>>();
            g_cpu_memory_pool = out;
        }
        return out;
    }

    template <>
    std::shared_ptr<MemoryPool<GPU>> MemoryPool<GPU>::instance()
    {
        auto out = g_gpu_memory_pool.lock();
        if (!out)
        {
            out = std::make_shared<MemoryPool<GPU>>();
            g_gpu_memory_pool = out;
        }
        return out;
    }

    template <>
    void MemoryPool<GPU>::setInstance(const std::shared_ptr<MemoryPool<GPU>>& pool)
    {
        g_gpu_memory_pool = pool;
    }

    template <>
    void MemoryPool<CPU>::setInstance(const std::shared_ptr<MemoryPool<CPU>>& pool)
    {
        g_cpu_memory_pool = pool;
    }

    template class PoolPolicy<CPU>;
    template class PoolPolicy<GPU>;
    template class MemoryPool<CPU>;
    template class MemoryPool<GPU>;
}
