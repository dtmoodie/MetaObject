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

    template<class CPUAllocator, class GPUAllocator>
    MergedAllocator<CPUAllocator, GPUAllocator>::MergedAllocator()
    {

    }

    template<class CPUAllocator, class GPUAllocator>
    unsigned char* MergedAllocator<CPUAllocator, GPUAllocator>::allocateGpu(size_t num_bytes, size_t element_size)
    {
        return GPUAllocator::allocate(num_bytes, element_size);
    }

    template<class CPUAllocator, class GPUAllocator>
    void MergedAllocator<CPUAllocator, GPUAllocator>::deallocateGpu(unsigned char* ptr, size_t num_bytes)
    {
        GPUAllocator::deallocate(ptr, num_bytes);
    }

    template<class CPUAllocator, class GPUAllocator>
    unsigned char* MergedAllocator<CPUAllocator, GPUAllocator>::allocateCpu(size_t num_bytes, size_t element_size)
    {
        return CPUAllocator::allocate(num_bytes, element_size);
    }

    template<class CPUAllocator, class GPUAllocator>
    void MergedAllocator<CPUAllocator, GPUAllocator>::deallocateCpu(unsigned char* ptr, size_t num_bytes)
    {
        CPUAllocator::deallocate(ptr, num_bytes);
    }
}
