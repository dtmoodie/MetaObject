#pragma once
#include "../Allocator.hpp"
namespace mo
{
    template<class CPUAllocator, class GPUAllocator>
    class MergedAllocator: 
        virtual public CPUAllocator, 
        virtual public GPUAllocator, 
        virtual public Allocator
    {
    public:
        MergedAllocator();
        virtual unsigned char* allocateGpu(size_t num_bytes, size_t element_size = 1) override;
        virtual void deallocateGpu(unsigned char* ptr, size_t numBytes) override;

        virtual unsigned char* allocateCpu(size_t num_bytes, size_t element_size = 1) override;
        virtual void deallocateCpu(unsigned char* ptr, size_t numBytes) override;
    };
}