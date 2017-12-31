#pragma once
#include "MetaObject/detail/Export.hpp"
#include <map>

namespace mo
{
    class MO_EXPORTS GPUMemory
    {
      protected:
        inline void _allocate(unsigned char** data, size_t size);
        inline void _deallocate(unsigned char* data);
    };

    class MO_EXPORTS CPUMemory
    {
      protected:
        inline void _allocate(unsigned char** data, size_t size);
        inline void _deallocate(unsigned char* data);
    };

    template <class XPU>
    class MO_EXPORTS MemoryBlock : public XPU
    {
      public:
        MemoryBlock(size_t size_);
        ~MemoryBlock();

        unsigned char* allocate(size_t size_, size_t elemSize_);
        bool deAllocate(unsigned char* ptr);
        const unsigned char* begin() const;
        const unsigned char* end() const;
        unsigned char* begin();
        unsigned char* end();
        size_t size() const;

      protected:
        unsigned char* m_begin;
        unsigned char* m_end;
        std::map<unsigned char*, unsigned char*> m_allocated_blocks;
    };

    typedef MemoryBlock<GPUMemory> GpuMemoryBlock;
    typedef MemoryBlock<CPUMemory> CpuMemoryBlock;
}
