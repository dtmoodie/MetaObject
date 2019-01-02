#ifndef MO_CUDA_CV_ALLOCATOR_HPP
#define MO_CUDA_CV_ALLOCATOR_HPP
#include "Allocator.hpp"

#include <MetaObject/core/detail/Allocator.hpp>
#include <MetaObject/core/detail/allocator_policies/Continuous.hpp>
#include <MetaObject/logging/logging.hpp>

#include <opencv2/core/cuda.hpp>

namespace mo
{
    namespace cuda
    {
        using GpuMat = cv::cuda::GpuMat;

        template<class PADDING_POLICY = mo::ContinuousPolicy>
        struct AllocatorProxy: public GpuMat::Allocator
        {
            inline AllocatorProxy(mo::Allocator* allocator);
            inline bool allocate(GpuMat* mat, int rows, int cols, size_t elemSize) override;
            inline void free(GpuMat* mat) override;
        private:
            mo::Allocator* m_allocator;
            PADDING_POLICY m_pad_policy;
        };

        template<class PADDING_POLICY>
        AllocatorProxy<PADDING_POLICY>::AllocatorProxy(mo::Allocator* allocator):
            m_allocator(allocator)
        {
        }

        template<class PADDING_POLICY>
        bool AllocatorProxy<PADDING_POLICY>::allocate(GpuMat* mat, int rows, int cols, size_t elem_size)
        {
            size_t size_needed, stride;
            m_pad_policy.sizeNeeded(rows, cols, elem_size, size_needed, stride);
            uint8_t* ptr = m_allocator->allocate(size_needed, elem_size);
            if (ptr == nullptr)
            {
                return false;
            }

            mat->data = ptr;
            mat->step = stride;
            mat->refcount = static_cast<int*>(cv::fastMalloc(sizeof(int)));
            return true;
        }

        template<class PADDING_POLICY>
        void AllocatorProxy<PADDING_POLICY>::free(GpuMat* mat)
        {
            m_allocator->deallocate(mat->datastart, mat->dataend - mat->datastart);
        }
    }
}

#endif // MO_CUDA_CV_ALLOCATOR_HPP
