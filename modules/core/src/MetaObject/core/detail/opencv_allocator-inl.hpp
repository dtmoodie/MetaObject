#pragma once

#include "opencv_allocator.hpp"

namespace mo
{

    /////////////////////////////////////////////////////////
    // CPU
    template<class Base, class PaddingPolicy>
    cv::UMatData* CvAllocator<CPU, Base, PaddingPolicy>::allocate(int dims,
        const int* sizes,
        int type,
        void* data,
        size_t* step,
        int flags,
        cv::UMatUsageFlags usageFlags) const
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
            unsigned char* ptr = const_cast<Base*>(static_cast<Base const*>(this))->allocate(total, CV_ELEM_SIZE(type));
            u->data = u->origdata = ptr;
        }

        return u;
    }
    
    template<class Base, class PaddingPolicy>
    bool CvAllocator<CPU, Base, PaddingPolicy>::allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const
    {
        return false;
    }

    template<class Base, class PaddingPolicy>
    void CvAllocator<CPU, Base, PaddingPolicy>::deallocate(cv::UMatData* u) const
    {
        if (!u) return;

        CV_Assert(u->urefcount >= 0);
        CV_Assert(u->refcount >= 0);

        if (u->refcount == 0)
        {
            if (!(u->flags & cv::UMatData::USER_ALLOCATED))
            {
                const_cast<Base*>(static_cast<Base const*>(this))->deallocate(u->origdata, u->size);
                u->origdata = 0;
            }

            delete u;
        }
    }

    template<class Base, class PaddingPolicy>
    unsigned char* CvAllocator<CPU, Base, PaddingPolicy>::allocate(size_t num_bytes, size_t elem_size)
    {
        return Base::allocate(num_bytes, elem_size);
    }

    template<class Base, class PaddingPolicy>
    void CvAllocator<CPU, Base, PaddingPolicy>::deallocate(unsigned char* ptr, size_t num_bytes)
    {
        Base::deallocate(ptr, num_bytes);
    }

    ///////////////////////////////////////////////////////////
    // GPU
    template<class Base, class PaddingPolicy>
    bool CvAllocator<GPU, Base, PaddingPolicy>::allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
    {
        size_t size_needed, stride;
        PaddingPolicy::sizeNeeded(rows, cols, static_cast<int>(elemSize), size_needed, stride);
        unsigned char* ptr = static_cast<Base*>(this)->allocate(size_needed, elemSize);
        if (ptr)
        {
            mat->refcount = static_cast<int*>(cv::fastMalloc(sizeof(int)));
            return true;
        }
        return false;
    }

    template<class Base, class PaddingPolicy>
    void CvAllocator<GPU, Base, PaddingPolicy>::free(cv::cuda::GpuMat* mat)
    {
        static_cast<Base*>(this)->deallocate(mat->datastart, mat->step * mat->rows);
        cv::fastFree(mat->refcount);
    }

    template<class Base, class PaddingPolicy>
    unsigned char* CvAllocator<GPU, Base, PaddingPolicy>::allocate(size_t num_bytes, size_t elem_size)
    {
        return Base::allocate(num_bytes, elem_size);
    }
    
    template<class Base, class PaddingPolicy>
    void CvAllocator<GPU, Base, PaddingPolicy>::deallocate(unsigned char* ptr, size_t num_bytes)
    {
        Base::deallocate(ptr, num_bytes);
    }
}