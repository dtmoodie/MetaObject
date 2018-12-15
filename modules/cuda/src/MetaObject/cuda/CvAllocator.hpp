#ifndef MO_CUDA_CV_ALLOCATOR_HPP
#define MO_CUDA_CV_ALLOCATOR_HPP
#include "Allocator.hpp"

#include <MetaObject/core/detail/allocator_policies/Continuous.hpp>
#include <MetaObject/logging/logging.hpp>

#include <opencv2/core/cuda.hpp>

namespace mo
{
    namespace cuda
    {
        using GpuMat = cv::cuda::GpuMat;

        template <class BASE_ALLOCATOR, class PADDING_POLICY = mo::ContinuousPolicy>
        struct CvAllocator : public GpuMat::Allocator
        {
            template <class... ARGS>
            CvAllocator(ARGS&&... args);

            CvAllocator(BASE_ALLOCATOR&& alloc = BASE_ALLOCATOR(), PADDING_POLICY&& pad = PADDING_POLICY());

            bool allocate(GpuMat* mat, int rows, int cols, size_t elemSize) override;
            void free(GpuMat* mat) override;

          private:
            PADDING_POLICY m_pad_policy;
            BASE_ALLOCATOR m_base_allocator;
        };

        template <class BASE_ALLOCATOR, class PADDING_POLICY>
        template <class... ARGS>
        CvAllocator<BASE_ALLOCATOR, PADDING_POLICY>::CvAllocator(ARGS&&... args)
            : m_base_allocator(std::forward<ARGS>(args)...)
        {
        }

        template <class BASE_ALLOCATOR, class PADDING_POLICY>
        CvAllocator<BASE_ALLOCATOR, PADDING_POLICY>::CvAllocator(BASE_ALLOCATOR&& alloc, PADDING_POLICY&& pad)
            : m_pad_policy(pad)
            , m_base_allocator(alloc)

        {
        }

        template <class BASE_ALLOCATOR, class PADDING_POLICY>
        bool CvAllocator<BASE_ALLOCATOR, PADDING_POLICY>::allocate(GpuMat* mat, int rows, int cols, size_t elem_size)
        {
            uint64_t size_needed, stride;
            m_pad_policy.sizeNeeded(rows, cols, elem_size, size_needed, stride);
            uint8_t* ptr = m_base_allocator.allocate(size_needed, elem_size);
            if (ptr == nullptr)
            {
                return false;
            }

            mat->data = ptr;
            mat->step = stride;
            mat->refcount = static_cast<int*>(cv::fastMalloc(sizeof(int)));
            return true;
        }

        template <class BASE_ALLOCATOR, class PADDING_POLICY>
        void CvAllocator<BASE_ALLOCATOR, PADDING_POLICY>::free(GpuMat* mat)
        {
            m_base_allocator.deallocate(mat->datastart, mat->dataend - mat->datastart);
        }
    }
}

#endif // MO_CUDA_CV_ALLOCATOR_HPP
