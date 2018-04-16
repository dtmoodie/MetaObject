#pragma once
#include "../MemoryBlock.hpp"
#include <ctime>
#include <list>
#include <unordered_map>
namespace mo
{
    template<class XPU>
    class StackPolicy
    {
    public:
        unsigned char* allocate(size_t num_bytes, size_t elem_size);

        void deallocate(unsigned char* ptr, size_t num_bytes);
        void release();
    private:
        void clear();
        struct FreeMemory
        {
            unsigned char* ptr;
            clock_t free_time;
            size_t size;
        };
        std::list<FreeMemory> m_deallocate_list;
        size_t m_deallocate_delay; // ms
    };

    using GpuStackPolicy = StackPolicy<GPU>;
    using CpuStackPolicy = StackPolicy<CPU>;
}