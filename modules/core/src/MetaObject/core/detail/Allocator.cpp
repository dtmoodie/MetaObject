#include "MetaObject/core/detail/AllocatorImpl.hpp"
#include "MetaObject/thread/cuda.hpp"
#include <MetaObject/core/SystemTable.hpp>
#include <RuntimeObjectSystem/ObjectInterfacePerModule.h>

#include "singleton.hpp"
#include <ctime>

namespace mo
{
    AllocationPolicy::~AllocationPolicy() {}

    thread_local std::string g_current_scope;

    const std::string& getScopeName() { return g_current_scope; }

    void setScopeName(const std::string& name) { g_current_scope = name; }

    class CpuMemoryPoolImpl : public CpuMemoryPool
    {
      public:
        CpuMemoryPoolImpl(size_t initial_size = 1e8) : m_total_usage(0), m_initial_block_size(initial_size)
        {
            m_blocks.emplace_back(std::make_unique<CpuMemoryBlock>(m_initial_block_size));
        }

        bool allocate(void** ptr_out, size_t total, size_t elemSize)
        {
            int index = 0;
            unsigned char* ptr;
            for (auto& block : m_blocks)
            {
                ptr = block->allocate(total, elemSize);
                if (ptr)
                {
                    *ptr_out = ptr;
                    return true;
                }
                ++index;
            }
            MO_LOG(trace) << "Creating new block of page locked memory for allocation.";
            m_blocks.push_back(std::make_unique<mo::CpuMemoryBlock>(std::max(m_initial_block_size / 2, total)));
            ptr = (*m_blocks.rbegin())->allocate(total, elemSize);
            if (ptr)
            {
                *ptr_out = ptr;
                return true;
            }
            return false;
        }

        uchar* allocate(size_t num_bytes)
        {
            int index = 0;
            unsigned char* ptr;
            for (auto& block : m_blocks)
            {
                ptr = block->allocate(num_bytes, sizeof(uchar));
                if (ptr)
                {
                    return ptr;
                }
                ++index;
            }
            m_blocks.push_back(std::make_unique<mo::CpuMemoryBlock>(std::max(m_initial_block_size / 2, num_bytes)));
            ptr = (*m_blocks.rbegin())->allocate(num_bytes, sizeof(uchar));
            if (ptr)
            {
                return ptr;
            }
            return nullptr;
        }

        bool deallocate(void* ptr, size_t total)
        {
            for (const auto& itr : m_blocks)
            {
                if (ptr >= itr->begin() && ptr < itr->end())
                {
                    if (itr->deAllocate(static_cast<unsigned char*>(ptr)))
                    {
                        m_total_usage -= total;
                        return true;
                    }
                }
            }
            return false;
        }

      private:
        size_t m_total_usage = 0;
        size_t m_initial_block_size;
        std::list<std::unique_ptr<mo::CpuMemoryBlock>> m_blocks;
    };

    class mt_CpuMemoryPoolImpl : public CpuMemoryPoolImpl
    {
      public:
        bool allocate(void** ptr, size_t total, size_t elemSize)
        {
            boost::mutex::scoped_lock lock(mtx);
            return CpuMemoryPoolImpl::allocate(ptr, total, elemSize);
        }
        uchar* allocate(size_t total)
        {
            boost::mutex::scoped_lock lock(mtx);
            return CpuMemoryPoolImpl::allocate(total);
        }

        bool deallocate(void* ptr, size_t total)
        {
            boost::mutex::scoped_lock lock(mtx);
            return CpuMemoryPoolImpl::deallocate(ptr, total);
        }

      private:
        boost::mutex mtx;
    };

    CpuMemoryPool* CpuMemoryPool::globalInstance()
    {
        static CpuMemoryPool* g_inst = nullptr;
        if (g_inst == nullptr)
        {
            g_inst = new mt_CpuMemoryPoolImpl();
        }
        return g_inst;
    }

    CpuMemoryPool* CpuMemoryPool::threadInstance()
    {
        static boost::thread_specific_ptr<CpuMemoryPool> g_inst;
        if (g_inst.get() == nullptr)
        {
            g_inst.reset(new CpuMemoryPoolImpl());
        }
        return g_inst.get();
    }

    class CpuMemoryStackImpl : public CpuMemoryStack
    {
      public:
        typedef cv::Mat MatType;
        CpuMemoryStackImpl(size_t delay) : deallocation_delay(delay) {}

        ~CpuMemoryStackImpl() { cleanup(true, true); }

        bool allocate(void** ptr, size_t total, size_t elemSize)
        {
            (void)elemSize;
            for (auto itr = deallocate_stack.begin(); itr != deallocate_stack.end(); ++itr)
            {
                if (std::get<2>(*itr) == total)
                {
                    *ptr = std::get<0>(*itr);
                    deallocate_stack.erase(itr);
                    return true;
                }
            }
            this->total_usage += total;
            CV_CUDEV_SAFE_CALL(cudaMallocHost(ptr, total));
            return true;
        }

        uchar* allocate(size_t total)
        {
            for (auto itr = deallocate_stack.begin(); itr != deallocate_stack.end(); ++itr)
            {
                if (std::get<2>(*itr) == total)
                {
                    deallocate_stack.erase(itr);
                    return std::get<0>(*itr);
                }
            }
            this->total_usage += total;
            uchar* ptr = nullptr;
            CV_CUDEV_SAFE_CALL(cudaMallocHost(&ptr, total));
            return ptr;
        }

        bool deallocate(void* ptr, size_t total)
        {
            deallocate_stack.emplace_back(static_cast<unsigned char*>(ptr), clock(), total);
            cleanup();
            return true;
        }

      private:
        void cleanup(bool force = false, bool destructor = false)
        {
            if (isCudaThread())
                return;
            auto time = clock();
            if (force)
                time = 0;
            for (auto itr = deallocate_stack.begin(); itr != deallocate_stack.end();)
            {
                if ((time - std::get<1>(*itr)) > deallocation_delay)
                {
                    total_usage -= std::get<2>(*itr);
                    MO_CUDA_ERROR_CHECK(cudaFreeHost(static_cast<void*>(std::get<0>(*itr))),
                                        "Error freeing " << std::get<2>(*itr) << " bytes of memory");
                    itr = deallocate_stack.erase(itr);
                }
                else
                {
                    ++itr;
                }
            }
        }
        size_t total_usage;
        size_t deallocation_delay;
        std::list<std::tuple<unsigned char*, clock_t, size_t>> deallocate_stack;
    };

    class mt_CpuMemoryStackImpl : public CpuMemoryStackImpl
    {
      public:
        mt_CpuMemoryStackImpl(size_t delay) : CpuMemoryStackImpl(delay) {}
        bool allocate(void** ptr, size_t total, size_t elemSize)
        {
            boost::mutex::scoped_lock lock(mtx);
            return CpuMemoryStackImpl::allocate(ptr, total, elemSize);
        }
        uchar* allocate(size_t total)
        {
            boost::mutex::scoped_lock lock(mtx);
            return CpuMemoryStackImpl::allocate(total);
        }

        bool deallocate(void* ptr, size_t total)
        {
            boost::mutex::scoped_lock lock(mtx);
            return CpuMemoryStackImpl::deallocate(ptr, total);
        }

      private:
        boost::mutex mtx;
    };

    CpuMemoryStack* CpuMemoryStack::globalInstance()
    {
        static CpuMemoryStack* g_inst = nullptr;
        if (g_inst == nullptr)
        {
            g_inst = new mt_CpuMemoryStackImpl(static_cast<size_t>(1.5 * CLOCKS_PER_SEC));
        }
        return g_inst;
    }

    CpuMemoryStack* CpuMemoryStack::threadInstance()
    {
        static boost::thread_specific_ptr<CpuMemoryStack> g_inst;
        if (g_inst.get() == nullptr)
        {
            g_inst.reset(new CpuMemoryStackImpl(static_cast<size_t>(1.5 * CLOCKS_PER_SEC)));
        }
        return g_inst.get();
    }

    std::shared_ptr<Allocator> Allocator::getThreadSafeAllocator()
    {
        return sharedSingleton<mt_UniversalAllocator_t>();
    }

    thread_local std::shared_ptr<Allocator> t_allocator;

    std::shared_ptr<Allocator> Allocator::getThreadSpecificAllocator()
    {
        if (t_allocator)
        {
            return t_allocator;
        }
        auto table = SystemTable::instance();
        if (!table->allocator)
        {
            table->allocator = sharedThreadSpecificSingleton<mt_UniversalAllocator_t>();
        }
        return table->allocator;
    }

    void Allocator::setThreadSpecificAllocator(const std::shared_ptr<Allocator>& allocator) { t_allocator = allocator; }

    // ================================================================
    // CpuStackPolicy
    cv::UMatData* CpuStackPolicy::allocate(
        int dims, const int* sizes, int type, void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const
    {
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
            void* ptr = 0;
            CpuMemoryStack::threadInstance()->allocate(&ptr, total, CV_ELEM_SIZE(type));

            u->data = u->origdata = static_cast<uchar*>(ptr);
        }

        return u;
    }
    uchar* CpuStackPolicy::allocate(size_t total) { return CpuMemoryStack::threadInstance()->allocate(total); }

    bool CpuStackPolicy::allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const
    {
        (void)data;
        (void)accessflags;
        (void)usageFlags;
        return false;
    }
    void CpuStackPolicy::deallocate(uchar* ptr, size_t total)
    {
        CpuMemoryStack::threadInstance()->deallocate(ptr, total);
    }

    void CpuStackPolicy::deallocate(cv::UMatData* u) const
    {
        if (!u)
            return;

        CV_Assert(u->urefcount >= 0);
        CV_Assert(u->refcount >= 0);

        if (u->refcount == 0)
        {
            if (!(u->flags & cv::UMatData::USER_ALLOCATED))
            {
                CpuMemoryStack::threadInstance()->deallocate(u->origdata, u->size);
                u->origdata = 0;
            }

            delete u;
        }
    }

    // ================================================================
    // mt_CpuStackPolicy

    cv::UMatData* mt_CpuStackPolicy::allocate(
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
            void* ptr = 0;
            CpuMemoryStack::globalInstance()->allocate(&ptr, total, CV_ELEM_SIZE(type));

            u->data = u->origdata = static_cast<uchar*>(ptr);
        }

        return u;
    }

    bool mt_CpuStackPolicy::allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const
    {
        (void)data;
        (void)accessflags;
        (void)usageFlags;
        return false;
    }

    void mt_CpuStackPolicy::deallocate(cv::UMatData* u) const
    {
        if (!u)
            return;

        CV_Assert(u->urefcount >= 0);
        CV_Assert(u->refcount >= 0);

        if (u->refcount == 0)
        {
            if (!(u->flags & cv::UMatData::USER_ALLOCATED))
            {
                CpuMemoryStack::globalInstance()->deallocate(u->origdata, u->size);
                u->origdata = 0;
            }

            delete u;
        }
    }

    // ================================================================
    // CpuPoolPolicy
    cv::UMatData* CpuPoolPolicy::allocate(
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
            void* ptr = 0;
            CpuMemoryPool::threadInstance()->allocate(&ptr, total, CV_ELEM_SIZE(type));

            u->data = u->origdata = static_cast<uchar*>(ptr);
        }

        return u;
    }

    bool CpuPoolPolicy::allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const
    {
        (void)data;
        (void)accessflags;
        (void)usageFlags;
        return false;
    }

    void CpuPoolPolicy::deallocate(cv::UMatData* u) const
    {
        if (!u)
            return;

        CV_Assert(u->urefcount >= 0);
        CV_Assert(u->refcount >= 0);

        if (u->refcount == 0)
        {
            if (!(u->flags & cv::UMatData::USER_ALLOCATED))
            {
                CpuMemoryPool::threadInstance()->deallocate(u->origdata, u->size);
                u->origdata = 0;
            }

            delete u;
        }
    }
    uchar* CpuPoolPolicy::allocate(size_t num_bytes) { return CpuMemoryPool::threadInstance()->allocate(num_bytes); }

    void CpuPoolPolicy::deallocate(uchar* ptr, size_t num_bytes)
    {
        (void)ptr;
        (void)num_bytes;
    }

    // ================================================================
    // mt_CpuPoolPolicy
    cv::UMatData* mt_CpuPoolPolicy::allocate(
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
            void* ptr = 0;
            CpuMemoryPool::globalInstance()->allocate(&ptr, total, CV_ELEM_SIZE(type));

            u->data = u->origdata = static_cast<uchar*>(ptr);
        }

        return u;
    }

    bool mt_CpuPoolPolicy::allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const
    {
        (void)data;
        (void)accessflags;
        (void)usageFlags;
        return false;
    }

    void mt_CpuPoolPolicy::deallocate(cv::UMatData* u) const
    {
        if (!u)
            return;

        CV_Assert(u->urefcount >= 0);
        CV_Assert(u->refcount >= 0);

        if (u->refcount == 0)
        {
            if (!(u->flags & cv::UMatData::USER_ALLOCATED))
            {
                CpuMemoryPool::globalInstance()->deallocate(u->origdata, u->size);
                u->origdata = 0;
            }

            delete u;
        }
    }

    cv::UMatData* PinnedAllocator::allocate(
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
            void* ptr = 0;
            cudaMallocHost(&ptr, total);

            u->data = u->origdata = static_cast<uchar*>(ptr);
        }

        return u;
    }

    bool PinnedAllocator::allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const
    {
        (void)data;
        (void)accessflags;
        (void)usageFlags;
        return false;
    }

    void PinnedAllocator::deallocate(cv::UMatData* u) const
    {
        if (!u)
            return;

        CV_Assert(u->urefcount >= 0);
        CV_Assert(u->refcount >= 0);

        if (u->refcount == 0)
        {
            if (!(u->flags & cv::UMatData::USER_ALLOCATED))
            {
                cudaFreeHost(u->origdata);
                u->origdata = 0;
            }

            delete u;
        }
    }
} // namespace mo
