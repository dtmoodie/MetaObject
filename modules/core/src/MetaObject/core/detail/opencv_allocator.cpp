#include "opencv_allocator.hpp"
#include "allocator_policies/Continuous.hpp"

namespace mo
{

    CvAllocatorProxy<CUDA>::CvAllocatorProxy(const std::shared_ptr<mo::Allocator>& allocator) : m_allocator(allocator)
    {
    }

    bool CvAllocatorProxy<CUDA>::allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
    {
        size_t size_needed, stride;
        ContinuousPolicy::sizeNeeded(rows, cols, static_cast<int>(elemSize), size_needed, stride);
        unsigned char* ptr = m_allocator->allocateGpu(size_needed, elemSize);
        if (ptr)
        {
            mat->refcount = static_cast<int*>(cv::fastMalloc(sizeof(int)));
            return true;
        }
        return false;
    }

    void CvAllocatorProxy<CUDA>::free(cv::cuda::GpuMat* mat)
    {
        m_allocator->deallocateGpu(mat->datastart, mat->rows * mat->step);
    }

    CvAllocatorProxy<CPU>::CvAllocatorProxy(const std::shared_ptr<mo::Allocator>& allocator) : m_allocator(allocator) {}

    cv::UMatData* CvAllocatorProxy<CPU>::allocate(
        int dims, const int* sizes, int type, void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const
    {
        (void)flags;
        (void)usageFlags;
        size_t total = CV_ELEM_SIZE(type);
        for (int i = dims - 1; i >= 0; i--)
        {
            if (step)
            {
                if (data && step[i] != CV_AUTOSTEP)
                {
                    CV_Assert(total <= step[i]);
                    total = step[i];
                }
                else
                {
                    step[i] = total;
                }
            }

            total *= size_t(sizes[i]);
        }

        cv::UMatData* u = new cv::UMatData(this);
        u->size = total;

        if (data)
        {
            u->data = u->origdata = static_cast<uchar*>(data);
            u->flags |= cv::UMatData::USER_ALLOCATED;
        }
        else
        {
            unsigned char* ptr = m_allocator->allocateCpu(total, CV_ELEM_SIZE(type));
            u->data = u->origdata = ptr;
        }

        return u;
    }

    bool CvAllocatorProxy<CPU>::allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const
    {
        return false;
    }

    void CvAllocatorProxy<CPU>::deallocate(cv::UMatData* u) const
    {
        if (!u)
            return;

        CV_Assert(u->urefcount >= 0);
        CV_Assert(u->refcount >= 0);

        if (u->refcount == 0)
        {
            if (!(u->flags & cv::UMatData::USER_ALLOCATED))
            {
                m_allocator->deallocateCpu(u->origdata, u->size);
                u->origdata = 0;
            }

            delete u;
        }
    }
}
