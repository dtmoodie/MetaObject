#include "MetaObject/Detail/AllocatorImpl.hpp"

using namespace mo;
boost::thread_specific_ptr<Allocator> thread_specific_allocator;

typedef PoolPolicy<cv::cuda::GpuMat, ContinuousPolicy>   d_TensorPoolAllocator_t;
typedef LockPolicy<d_TensorPoolAllocator_t>              d_mt_TensorPoolAllocator_t;

typedef StackPolicy<cv::cuda::GpuMat, ContinuousPolicy>  d_TensorAllocator_t;
typedef StackPolicy<cv::cuda::GpuMat, PitchedPolicy>     d_TextureAllocator_t;

typedef LockPolicy<d_TensorAllocator_t>                  d_mt_TensorAllocator_t;
typedef LockPolicy<d_TextureAllocator_t>                 d_mt_TextureAllocator_t;

typedef PoolPolicy<cv::Mat, ContinuousPolicy>            h_PoolAllocator_t;
typedef StackPolicy<cv::Mat, ContinuousPolicy>           h_StackAllocator_t;

typedef LockPolicy<h_PoolAllocator_t>                    h_mt_PoolAllocator_t;
typedef LockPolicy<h_StackAllocator_t>                   h_mt_StackAllocator_t;

typedef CombinedPolicy<d_TensorPoolAllocator_t, d_TextureAllocator_t> d_UniversalAllocator_t;
typedef LockPolicy<d_UniversalAllocator_t> d_mt_UniversalAllocator_t;

typedef CombinedPolicy<h_PoolAllocator_t, h_StackAllocator_t> h_UniversalAllocator_t;
typedef LockPolicy<h_UniversalAllocator_t> h_mt_UniversalAllocator_t;


typedef ConcreteAllocator<h_mt_PoolAllocator_t, d_mt_TensorPoolAllocator_t> mt_TensorAllocator_t;
typedef ConcreteAllocator<h_PoolAllocator_t, d_TensorAllocator_t>           TensorAllocator_t;

typedef ConcreteAllocator<h_mt_StackAllocator_t, d_mt_TextureAllocator_t>   mt_TextureAllocator_t;
typedef ConcreteAllocator<h_StackAllocator_t, d_TextureAllocator_t>         TextureAllocator_t;

typedef ConcreteAllocator<h_UniversalAllocator_t, d_UniversalAllocator_t>    UniversalAllocator_t;
typedef ConcreteAllocator<h_mt_UniversalAllocator_t, d_mt_UniversalAllocator_t>    mt_UniversalAllocator_t;


class CpuMemoryPoolImpl: public CpuMemoryPool
{
public:
    bool allocate(void** ptr, size_t total, size_t elemSize)
    {
        int index = 0;
        unsigned char* _ptr;
        for (auto& block : blocks)
        {
            _ptr = block->allocate(total, elemSize);
            if (_ptr)
            {
                *ptr = _ptr;
                LOG(trace) << "Allocating " << total << " bytes from pre-allocated memory block number "
                           << index << " at address: " << (void*)_ptr;
                return true;
            }
            ++index;
        }
        LOG(trace) << "Creating new block of page locked memory for allocation.";
        blocks.push_back(
            std::shared_ptr<mo::CpuMemoryBlock>(
                new mo::CpuMemoryBlock(std::max(_initial_block_size / 2, total))));
        _ptr = (*blocks.rbegin())->allocate(total, elemSize);
        if (_ptr)
        {
            LOG(debug) << "Allocating " << total
                       << " bytes from newly created memory block at address: " << (void*)_ptr;
            *ptr = _ptr;
            return true;
        }
        return false;
    }

    bool deallocate(void* ptr, size_t total)
    {
        for (auto itr : blocks)
        {
            if (ptr > itr->Begin() && ptr < itr->End())
            {
                LOG(trace) << "Releasing memory block of size "
                                 << total << " at address: " << ptr;
                if (itr->deAllocate((unsigned char*)ptr))
                {
                    return true;
                }
            }
        }
        return false;
    }
private:
    size_t total_usage;
    size_t _initial_block_size;
    std::list<std::shared_ptr<mo::CpuMemoryBlock>> blocks;
};

CpuMemoryPool* CpuMemoryPool::GlobalInstance()
{
    static CpuMemoryPool* g_inst = nullptr;
    if(g_inst == nullptr)
    {
        g_inst = new CpuMemoryPoolImpl();
    }
    return g_inst;
}

CpuMemoryPool* CpuMemoryPool::ThreadInstance()
{
    static boost::thread_specific_ptr<CpuMemoryPool> g_inst;
    if(g_inst.get() == nullptr)
    {
        g_inst.reset(new CpuMemoryPoolImpl());
    }
    return g_inst.get();
}

class CpuMemoryStackImpl: public CpuMemoryStack
{
public:
    CpuMemoryStackImpl(size_t delay):
        deallocation_delay(delay) {}

    ~CpuMemoryStackImpl()
    {
        cleanup(true);
    }

    bool allocate(void** ptr, size_t total, size_t elemSize)
    {
        for (auto itr = deallocate_stack.begin(); itr != deallocate_stack.end(); ++itr)
        {
            if(std::get<2>(*itr) == total)
            {
                *ptr = std::get<0>(*itr);
                deallocate_stack.erase(itr);
                LOG(trace) << "[CPU] Reusing memory block of size "
                           << total / (1024 * 1024) << " MB. Total usage: "
                           << total_usage /(1024*1024) << " MB";
                return true;
            }
        }
        LOG(trace) << "[CPU] Allocating block of size "
                   << total / (1024 * 1024) << " MB. Total usage: "
                   << total_usage / (1024 * 1024) << " MB";
        CV_CUDEV_SAFE_CALL(cudaMallocHost(ptr, total));
        return true;
    }

    bool deallocate(void* ptr, size_t total)
    {
        LOG(trace) << "Releasing " << total / (1024 * 1024) << " MB to lazy deallocation pool";
        deallocate_stack.emplace_back((unsigned char*)ptr, clock(), total);
        cleanup();
        return true;
    }
private:
    void cleanup(bool force  = false)
    {
        auto time = clock();
        if (force)
            time = 0;
        for (auto itr = deallocate_stack.begin(); itr != deallocate_stack.end();)
        {
            if((time - std::get<1>(*itr)) > deallocation_delay)
            {
                total_usage -= std::get<2>(*itr);
                LOG(trace) << "[CPU] DeAllocating block of size " << std::get<2>(*itr) / (1024 * 1024)
                    << " MB. Which was stale for " << time - std::get<1>(*itr)
                    << " ms. Total usage: " << total_usage / (1024 * 1024) << " MB";
                CV_CUDEV_SAFE_CALL(cudaFreeHost((void*)std::get<0>(*itr)));
                itr = deallocate_stack.erase(itr);
            }else
            {
                ++itr;
            }
        }
    }
    size_t total_usage;
    size_t deallocation_delay;
    std::list<std::tuple<unsigned char*, clock_t, size_t>> deallocate_stack;
};

CpuMemoryStack* CpuMemoryStack::GlobalInstance()
{
    static CpuMemoryStack* g_inst = nullptr;
    if(g_inst == nullptr)
    {
        g_inst = new CpuMemoryStackImpl(1000);
    }
    return g_inst;
}

CpuMemoryStack* CpuMemoryStack::ThreadInstance()
{
    static boost::thread_specific_ptr<CpuMemoryStack> g_inst;
    if(g_inst.get() == nullptr)
    {
        g_inst.reset(new CpuMemoryStackImpl(1000));
    }
    return g_inst.get();
}

Allocator* Allocator::GetThreadSafeAllocator()
{
    static Allocator* g_inst = nullptr;
    if(g_inst == nullptr)
    {
        g_inst = new mt_UniversalAllocator_t();
    }
    return nullptr;
}

Allocator* Allocator::GetThreadSpecificAllocator()
{
    if(thread_specific_allocator.get() == nullptr)
    {
        thread_specific_allocator.reset(new UniversalAllocator_t());
    }
    return thread_specific_allocator.get();
}


cv::UMatData* StackPolicy<cv::Mat, ContinuousPolicy>::allocate(int dims, const int* sizes, int type,
    void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const
{
    return nullptr;
}

bool StackPolicy<cv::Mat, ContinuousPolicy>::allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const
{
    return false;
}

void StackPolicy<cv::Mat, ContinuousPolicy>::deallocate(cv::UMatData* data) const
{

}

cv::UMatData* PoolPolicy<cv::Mat, ContinuousPolicy>::allocate(int dims, const int* sizes, int type,
    void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const
{
    return nullptr;
}

bool PoolPolicy<cv::Mat, ContinuousPolicy>::allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const
{
    return false;
}

void PoolPolicy<cv::Mat, ContinuousPolicy>::deallocate(cv::UMatData* data) const
{

}
