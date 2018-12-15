#ifndef MO_CORE_ALLOCATOR_POLICIES_OPENCV_HPP
#define MO_CORE_ALLOCATOR_POLICIES_OPENCV_HPP
#include <forward_list>

#include <opencv2/core.hpp>

namespace mo
{
    template <class BASE_ALLOCATOR>
    struct CvAllocator : public cv::MatAllocator
    {
        template <class... ARGS>
        CvAllocator(ARGS&&... args);

        cv::UMatData* allocate(int dims,
                               const int* sizes,
                               int type,
                               void* data,
                               size_t* step,
                               int flags,
                               cv::UMatUsageFlags usageFlags) const override;
        bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const override;
        void deallocate(cv::UMatData* data) const override;

      private:
        mutable BASE_ALLOCATOR m_allocator;
    };

    ////////////////////////////////////////////////////////////////////////////////////
    ///              IMPLEMENTATION
    ////////////////////////////////////////////////////////////////////////////////////

    template <class BASE_ALLOCATOR>
    template <class... ARGS>
    CvAllocator<BASE_ALLOCATOR>::CvAllocator(ARGS&&... args)
        : m_allocator(std::forward<ARGS>(args)...)
    {
    }

    template <class BASE_ALLOCATOR>
    cv::UMatData* CvAllocator<BASE_ALLOCATOR>::allocate(
        int dims, const int* sizes, int type, void* data, size_t* step, int, cv::UMatUsageFlags) const
    {
        size_t total = static_cast<size_t>(CV_ELEM_SIZE(type));
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

            total *= static_cast<size_t>(sizes[i]);
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
            uint8_t* ptr = m_allocator.allocate(total, CV_ELEM_SIZE(type));
            CV_Assert(ptr);
            u->data = u->origdata = static_cast<uchar*>(ptr);
        }
    }

    template <class BASE_ALLOCATOR>
    bool CvAllocator<BASE_ALLOCATOR>::allocate(cv::UMatData*, int, cv::UMatUsageFlags) const
    {
        return false;
    }

    template <class BASE_ALLOCATOR>
    void CvAllocator<BASE_ALLOCATOR>::deallocate(cv::UMatData* data) const
    {
        if (!data)
        {

            return;
        }

        CV_Assert(data->urefcount >= 0);
        CV_Assert(data->refcount >= 0);

        if (data->refcount == 0)
        {
            if (!(data->flags & cv::UMatData::USER_ALLOCATED))
            {
                m_allocator.deallocate(data->origdata, data->size);
                data->origdata = nullptr;
            }
            delete data;
        }
    }
}

#endif // MO_CORE_ALLOCATOR_POLICIES_OPENCV_HPP
