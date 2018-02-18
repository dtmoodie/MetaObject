#pragma once
#include <MetaObject/Python.hpp>
#include <MetaObject/core/detail/Allocator.hpp>
#include <Python.h>

namespace mo
{
    void setupAllocator();
    class NumpyAllocator : virtual public Allocator
    {
      public:
        NumpyAllocator(std::shared_ptr<Allocator> default_allocator_ = Allocator::getThreadSafeAllocator());
        ~NumpyAllocator();

        cv::UMatData* allocate(PyObject* o, int dims, const int* sizes, int type, size_t* step) const;

        virtual cv::UMatData* allocate(int dims0,
                                       const int* sizes,
                                       int type,
                                       void* data,
                                       size_t* step,
                                       int flags,
                                       cv::UMatUsageFlags usageFlags) const override;

        virtual bool allocate(cv::UMatData* u, int accessFlags, cv::UMatUsageFlags usageFlags) const override;

        void deallocate(cv::UMatData* u) const override;

        // Used for stl allocators
        virtual unsigned char* allocateGpu(size_t num_bytes) override;
        virtual void deallocateGpu(uchar* ptr, size_t numBytes) override;

        virtual unsigned char* allocateCpu(size_t num_bytes) override;
        virtual void deallocateCpu(uchar* ptr, size_t numBytes) override;

        virtual bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize) override;
        virtual void free(cv::cuda::GpuMat* mat) override;

        std::shared_ptr<Allocator> default_allocator;
    };
}
