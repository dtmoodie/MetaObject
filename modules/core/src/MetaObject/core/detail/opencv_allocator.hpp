#pragma once
#include "MemoryBlock.hpp"
#include <opencv2/core/cuda.hpp>
namespace mo
{
    template<class XPU, class Base, class PaddingPolicy>
    class CvAllocator;

    template<class Base, class PaddingPolicy>
    class CvAllocator<CPU, Base, PaddingPolicy> : 
        virtual public cv::MatAllocator, 
        virtual public Base, 
        virtual public PaddingPolicy
    {
    public:
        typedef cv::Mat MatType;
        
        virtual cv::UMatData* allocate(int dims,
            const int* sizes,
            int type,
            void* data,
            size_t* step,
            int flags,
            cv::UMatUsageFlags usageFlags) const override;
        virtual bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const override;
        virtual void deallocate(cv::UMatData* data) const override;

        unsigned char* allocate(size_t num_bytes, size_t elem_size);
        void deallocate(unsigned char* ptr, size_t num_bytes);
    };

    template<class Base, class PaddingPolicy>
    class CvAllocator<GPU, Base, PaddingPolicy>: 
        virtual public cv::cuda::GpuMat::Allocator, 
        virtual public Base,
        virtual public PaddingPolicy

    {
    public:
        typedef cv::cuda::GpuMat MatType;
        virtual bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize) override;
        virtual void free(cv::cuda::GpuMat* mat) override;

        unsigned char* allocate(size_t num_bytes, size_t elem_size);
        void deallocate(unsigned char* ptr, size_t num_bytes);
    };
}