#pragma once
#include "MetaObject/detail/Export.hpp"
#include <unordered_map>

namespace mo
{
    struct MO_EXPORTS CPU
    {
        static void allocate(unsigned char** data, size_t size);
        static void deallocate(unsigned char* data);
    };

    struct MO_EXPORTS CUDA
    {
        static void allocate(unsigned char** data, size_t size);
        static void deallocate(unsigned char* data);
    };

    using GPU = CUDA;

    template <class XPU>
    class MO_EXPORTS Memory
    {
      public:
        static void allocate(unsigned char** data, size_t size)
        {
            XPU::allocate(data, size);
        }
        static void deallocate(unsigned char* data)
        {
            XPU::deallocate(data);
        }
    };

    template <class XPU>
    class MO_EXPORTS MemoryBlock
    {
      public:
        MemoryBlock(size_t size_);
        ~MemoryBlock();

        unsigned char* allocate(size_t size_, size_t elemSize_);
        bool deAllocate(unsigned char* ptr, size_t size);
        const unsigned char* begin() const;
        const unsigned char* end() const;
        unsigned char* begin();
        unsigned char* end();
        size_t size() const;

      protected:
        unsigned char* m_begin;
        unsigned char* m_end;
        std::unordered_map<unsigned char*, unsigned char*> m_allocated_blocks;
    };

    using GPUMemory = Memory<CUDA>;
    using CPUMemory = Memory<CPU>;
    typedef MemoryBlock<GPU> GpuMemoryBlock;
    typedef MemoryBlock<CPU> CpuMemoryBlock;
}
