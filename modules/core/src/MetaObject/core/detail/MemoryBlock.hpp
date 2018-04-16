#pragma once
#include "MetaObject/detail/Export.hpp"
#include <unordered_map>

namespace mo
{
    struct CPU;
    struct GPU;

    template<class XPU> 
    class MO_EXPORTS Memory
    {
    public:
        static void allocate(unsigned char** data, size_t size);
        static void deallocate(unsigned char* data);
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

    using GPUMemory = Memory<GPU>;
    using CPUMemory = Memory<CPU>;
    typedef MemoryBlock<GPU> GpuMemoryBlock;
    typedef MemoryBlock<CPU> CpuMemoryBlock;
}
