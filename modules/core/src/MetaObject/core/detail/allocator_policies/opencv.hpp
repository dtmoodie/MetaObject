#ifndef MO_CORE_ALLOCATOR_POLICIES_OPENCV_HPP
#define MO_CORE_ALLOCATOR_POLICIES_OPENCV_HPP
#include <MetaObject/core/detail/Allocator.hpp>

#include <forward_list>
#include <opencv2/core.hpp>

namespace mo
{
    struct Allocator;

    // Todo add backward compat here
#if CV_VERSION_MAJOR < 4
    using AccessFlag = int;
#else
    using AccessFlag = cv::AccessFlag;
#endif
    const constexpr size_t AUTO_STEP = cv::Mat::AUTO_STEP;

    struct CvAllocatorProxy : public cv::MatAllocator
    {
        inline CvAllocatorProxy(mo::Allocator* allocator)
            : m_allocator(allocator)
        {
        }

        ~CvAllocatorProxy()
        {
            MO_LOG(debug, "Opencv allocator destroyed");
        }

        inline cv::UMatData* allocate(int dims,
                                      const int* sizes,
                                      int type,
                                      void* data,
                                      size_t* step,
                                      AccessFlag,
                                      cv::UMatUsageFlags) const override
        {
            auto total = static_cast<size_t>(CV_ELEM_SIZE(type));
            for (int i = dims - 1; i >= 0; i--)
            {
                if (step)
                {
                    if (data && step[i] != AUTO_STEP)
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

            auto u = new cv::UMatData(this);
            u->size = total;

            if (data)
            {
                u->data = u->origdata = static_cast<uchar*>(data);
                u->flags |= cv::UMatData::USER_ALLOCATED;
            }
            else
            {
                uint8_t* ptr = m_allocator->allocate(total, CV_ELEM_SIZE(type));
                CV_Assert(ptr);
                u->data = u->origdata = static_cast<uchar*>(ptr);
            }
            return u;
        }

        inline bool allocate(cv::UMatData*, AccessFlag, cv::UMatUsageFlags) const override
        {
            return false;
        }

        inline void deallocate(cv::UMatData* data) const override
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
                    m_allocator->deallocate(data->origdata, data->size);
                    data->origdata = nullptr;
                }
                delete data;
            }
        }

      private:
        mutable mo::Allocator* m_allocator;
    };

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
                               AccessFlag flags,
                               cv::UMatUsageFlags usageFlags) const override;
        bool allocate(cv::UMatData* data, AccessFlag accessflags, cv::UMatUsageFlags usageFlags) const override;
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
        int dims, const int* sizes, int type, void* data, size_t* step, AccessFlag, cv::UMatUsageFlags) const
    {
        auto total = static_cast<size_t>(CV_ELEM_SIZE(type));
        for (int i = dims - 1; i >= 0; i--)
        {
            if (step)
            {
                if (data && step[i] != AUTO_STEP)
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

        auto u = new cv::UMatData(this);
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
        return u;
    }

    template <class BASE_ALLOCATOR>
    bool CvAllocator<BASE_ALLOCATOR>::allocate(cv::UMatData*, AccessFlag, cv::UMatUsageFlags) const
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
} // namespace mo

#endif // MO_CORE_ALLOCATOR_POLICIES_OPENCV_HPP
