#include "MetaObject/core/detail/AllocatorImpl.hpp"
#include "MetaObject/thread/cuda.hpp"
#include <ctime>

using namespace mo;
AllocationPolicy::~AllocationPolicy(){}
boost::thread_specific_ptr<Allocator> thread_specific_allocator;
thread_local Allocator* thread_specific_allocator_unowned = nullptr;
thread_local std::string current_scope;

thread_local cv::MatAllocator* t_cpuAllocator = nullptr;
cv::MatAllocator* g_cpuAllocator = nullptr;

thread_local cv::cuda::GpuMat::Allocator* t_gpuAllocator = nullptr;
cv::cuda::GpuMat::Allocator* g_gpuAllocator = nullptr;

cv::UMatData* CpuAllocatorThreadAdapter::allocate(int dims, const int* sizes, int type,
        void* data, size_t* step, int flags,
        cv::UMatUsageFlags usageFlags) const {
    if(t_cpuAllocator) {
        return t_cpuAllocator->allocate(dims, sizes, type, data, step, flags, usageFlags);
    } else {
        if(g_cpuAllocator == nullptr) {
            g_cpuAllocator = mo::Allocator::getThreadSafeAllocator();
        }
        return g_cpuAllocator->allocate(dims, sizes, type, data, step, flags, usageFlags);
    }
}

bool CpuAllocatorThreadAdapter::allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const {
    if(t_cpuAllocator) {
        return t_cpuAllocator->allocate(data, accessflags, usageFlags);
    } else {
        if(g_cpuAllocator == nullptr) {
            g_cpuAllocator = mo::Allocator::getThreadSafeAllocator();
        }
        return g_cpuAllocator->allocate(data, accessflags, usageFlags);
    }
}

void CpuAllocatorThreadAdapter::deallocate(cv::UMatData* data) const {
    if(t_cpuAllocator) {
        t_cpuAllocator->deallocate(data);
    } else {
        if(g_cpuAllocator == nullptr) {
            g_cpuAllocator = mo::Allocator::getThreadSafeAllocator();
        }
        g_cpuAllocator->deallocate(data);
    }
}

void CpuAllocatorThreadAdapter::setThreadAllocator(cv::MatAllocator* allocator) {
    t_cpuAllocator = allocator;
}

void CpuAllocatorThreadAdapter::setGlobalAllocator(cv::MatAllocator* allocator) {
    g_cpuAllocator = allocator;
}

bool GpuAllocatorThreadAdapter::allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize) {
    if(t_gpuAllocator) {
        return t_gpuAllocator->allocate(mat, rows, cols, elemSize);
    } else {
        if(g_gpuAllocator == nullptr) {
            g_gpuAllocator = mo::Allocator::getThreadSafeAllocator();
        }
        return g_gpuAllocator->allocate(mat, rows, cols, elemSize);
    }
}


void GpuAllocatorThreadAdapter::free(cv::cuda::GpuMat* mat) {
    if(t_gpuAllocator) {
        t_gpuAllocator->free(mat);
    } else {
        if(g_gpuAllocator == nullptr) {
            g_gpuAllocator = mo::Allocator::getThreadSafeAllocator();
        }
        g_gpuAllocator->free(mat);
    }
}

void GpuAllocatorThreadAdapter::setThreadAllocator(cv::cuda::GpuMat::Allocator* allocator) {
    t_gpuAllocator = allocator;
}

void GpuAllocatorThreadAdapter::setGlobalAllocator(cv::cuda::GpuMat::Allocator* allocator) {
    g_gpuAllocator = allocator;
}

void mo::setScopeName(const std::string& name) {
    current_scope = name;
}

const std::string& mo::getScopeName() {
    return current_scope;
}

class CpuMemoryPoolImpl: public CpuMemoryPool {
public:
    CpuMemoryPoolImpl(size_t initial_size = 1e8):
        total_usage(0),
        _initial_block_size(initial_size){
        blocks.emplace_back(new mo::CpuMemoryBlock(_initial_block_size));
    }

    bool allocate(void** ptr, size_t total, size_t elemSize) {
        int index = 0;
        unsigned char* _ptr;
        for (auto& block : blocks) {
            _ptr = block->allocate(total, elemSize);
            if (_ptr) {
                *ptr = _ptr;
#if defined(_DEBUG) && defined(DEBUG_ALLOCATION)
                MO_LOG(trace) << "Allocating " << total << " bytes from pre-allocated memory block number "
                           << index << " at address: " << static_cast<void*>(_ptr);
#endif
                return true;
            }
            ++index;
        }
        MO_LOG(trace) << "Creating new block of page locked memory for allocation.";
        blocks.push_back(
            std::shared_ptr<mo::CpuMemoryBlock>(
                new mo::CpuMemoryBlock(std::max(_initial_block_size / 2, total))));
        _ptr = (*blocks.rbegin())->allocate(total, elemSize);
        if (_ptr) {
#if defined(_DEBUG) && defined(DEBUG_ALLOCATION)
            MO_LOG(debug) << "Allocating " << total
                       << " bytes from newly created memory block at address: " << static_cast<void*>(_ptr);
#endif
            *ptr = _ptr;
            return true;
        }
        return false;
    }
    uchar* allocate(size_t num_bytes) {
        int index = 0;
        unsigned char* _ptr;
        for (auto& block : blocks) {
            _ptr = block->allocate(num_bytes, sizeof(uchar));
            if (_ptr) {
#if defined(_DEBUG) && defined(DEBUG_ALLOCATION)
                MO_LOG(trace) << "Allocating " << num_bytes << " bytes from pre-allocated memory block number "
                           << index << " at address: " << static_cast<void*>(_ptr);
#endif
                return _ptr;
            }
            ++index;
        }
#if defined(_DEBUG) && defined(DEBUG_ALLOCATION)
        MO_LOG(trace) << "Creating new block of page locked memory for allocation.";
#endif
        blocks.push_back(
            std::shared_ptr<mo::CpuMemoryBlock>(
                new mo::CpuMemoryBlock(std::max(_initial_block_size / 2, num_bytes))));
        _ptr = (*blocks.rbegin())->allocate(num_bytes, sizeof(uchar));
        if (_ptr) {
#if defined(_DEBUG) && defined(DEBUG_ALLOCATION)
            MO_LOG(debug) << "Allocating " << num_bytes
                       << " bytes from newly created memory block at address: " << static_cast<void*>(_ptr);
#endif
            return _ptr;
        }
        return nullptr;
    }
    bool deallocate(void* ptr, size_t total) {
        for (auto itr : blocks) {
            if (ptr >= itr->Begin() && ptr < itr->End()) {
#if defined(_DEBUG) && defined(DEBUG_ALLOCATION)
                MO_LOG(trace) << "Releasing memory block of size "
                           << total << " at address: " << ptr;
#endif
                if (itr->deAllocate(static_cast<unsigned char*>(ptr))) {
                    return true;
                }
            }
        }
        return false;
    }
private:
    size_t total_usage = 0;
    size_t _initial_block_size;
    std::list<std::shared_ptr<mo::CpuMemoryBlock>> blocks;
};

class mt_CpuMemoryPoolImpl: public CpuMemoryPoolImpl {
public:
    bool allocate(void** ptr, size_t total, size_t elemSize) {
        boost::mutex::scoped_lock lock(mtx);
        return CpuMemoryPoolImpl::allocate(ptr, total, elemSize);
    }
    uchar* allocate(size_t total) {
        boost::mutex::scoped_lock lock(mtx);
        return CpuMemoryPoolImpl::allocate(total);
    }

    bool deallocate(void* ptr, size_t total) {
        boost::mutex::scoped_lock lock(mtx);
        return CpuMemoryPoolImpl::deallocate(ptr, total);
    }
private:
    boost::mutex mtx;
};
CpuMemoryPool* CpuMemoryPool::globalInstance() {
    static CpuMemoryPool* g_inst = nullptr;
    if(g_inst == nullptr) {
        g_inst = new mt_CpuMemoryPoolImpl();
    }
    return g_inst;
}

CpuMemoryPool* CpuMemoryPool::threadInstance() {
    static boost::thread_specific_ptr<CpuMemoryPool> g_inst;
    if(g_inst.get() == nullptr) {
        g_inst.reset(new CpuMemoryPoolImpl());
    }
    return g_inst.get();
}

class CpuMemoryStackImpl: public CpuMemoryStack {
public:
    typedef cv::Mat MatType;
    CpuMemoryStackImpl(size_t delay):
        deallocation_delay(delay) {}

    ~CpuMemoryStackImpl() {
        cleanup(true, true);
    }

    bool allocate(void** ptr, size_t total, size_t elemSize) {
        (void)elemSize;
        for (auto itr = deallocate_stack.begin(); itr != deallocate_stack.end(); ++itr) {
            if(std::get<2>(*itr) == total) {
                *ptr = std::get<0>(*itr);
                deallocate_stack.erase(itr);
#if defined(_DEBUG) && defined(DEBUG_ALLOCATION)
                MO_LOG(trace) << "[CPU] Reusing memory block of size "
                           << total / (1024 * 1024) << " MB. Total usage: "
                           << total_usage /(1024*1024) << " MB";
#endif
                return true;
            }
        }
        this->total_usage += total;
#if defined(_DEBUG) && defined(DEBUG_ALLOCATION)
        MO_LOG(trace) << "[CPU] Allocating block of size "
                   << total / (1024 * 1024) << " MB. Total usage: "
                   << total_usage / (1024 * 1024) << " MB";
#endif
        CV_CUDEV_SAFE_CALL(cudaMallocHost(ptr, total));
        return true;
    }
    uchar* allocate(size_t total) {
        for (auto itr = deallocate_stack.begin(); itr != deallocate_stack.end(); ++itr) {
            if(std::get<2>(*itr) == total) {

                deallocate_stack.erase(itr);
#if defined(_DEBUG) && defined(DEBUG_ALLOCATION)
                MO_LOG(trace) << "[CPU] Reusing memory block of size "
                           << total / (1024 * 1024) << " MB. Total usage: "
                           << total_usage /(1024*1024) << " MB";
#endif
                return std::get<0>(*itr);
            }
        }
        this->total_usage += total;
#if defined(_DEBUG) && defined(DEBUG_ALLOCATION)
        MO_LOG(trace) << "[CPU] Allocating block of size "
                   << total / (1024 * 1024) << " MB. Total usage: "
                   << total_usage / (1024 * 1024) << " MB";
#endif
        uchar* ptr = nullptr;
        CV_CUDEV_SAFE_CALL(cudaMallocHost(&ptr, total));
        return ptr;
    }

    bool deallocate(void* ptr, size_t total) {
#if defined(_DEBUG) && defined(DEBUG_ALLOCATION)
        MO_LOG(trace) << "Releasing " << total / (1024 * 1024) << " MB to lazy deallocation pool";
#endif
        deallocate_stack.emplace_back(static_cast<unsigned char*>(ptr), clock(), total);
        cleanup();
        return true;
    }
private:
    void cleanup(bool force  = false, bool destructor = false) {
        if(isCudaThread())
            return;
        auto time = clock();
        if (force)
            time = 0;
        for (auto itr = deallocate_stack.begin(); itr != deallocate_stack.end();) {
            if((time - std::get<1>(*itr)) > deallocation_delay) {
                total_usage -= std::get<2>(*itr);
                if(!destructor) {
#if defined(_DEBUG) && defined(DEBUG_ALLOCATION)
#ifdef _MSC_VER
                    MO_LOG(trace) << "[CPU] DeAllocating block of size " << std::get<2>(*itr) / (1024 * 1024)
                               << " MB. Which was stale for " << time - std::get<1>(*itr)
                               << " ms. Total usage: " << total_usage / (1024 * 1024) << " MB";
#else
                    MO_LOG(trace) << "[CPU] DeAllocating block of size " << std::get<2>(*itr) / (1024 * 1024)
                               << " MB. Which was stale for " << (time - std::get<1>(*itr)) / 1000
                               << " ms. Total usage: " << total_usage / (1024 * 1024) << " MB";
#endif
#endif
                }
                MO_CUDA_ERROR_CHECK(cudaFreeHost(static_cast<void*>(std::get<0>(*itr))), "Error freeing " << std::get<2>(*itr) << " bytes of memory");
                itr = deallocate_stack.erase(itr);
            } else {
                ++itr;
            }
        }
    }
    size_t total_usage;
    size_t deallocation_delay;
    std::list<std::tuple<unsigned char*, clock_t, size_t>> deallocate_stack;
};

class mt_CpuMemoryStackImpl: public CpuMemoryStackImpl {
public:
    mt_CpuMemoryStackImpl(size_t delay):
        CpuMemoryStackImpl(delay) {}
    bool allocate(void** ptr, size_t total, size_t elemSize) {
        boost::mutex::scoped_lock lock(mtx);
        return CpuMemoryStackImpl::allocate(ptr, total, elemSize);
    }
    uchar* allocate(size_t total) {
        boost::mutex::scoped_lock lock(mtx);
        return CpuMemoryStackImpl::allocate(total);
    }

    bool deallocate(void* ptr, size_t total) {
        boost::mutex::scoped_lock lock(mtx);
        return CpuMemoryStackImpl::deallocate(ptr, total);
    }
private:
    boost::mutex mtx;
};

CpuMemoryStack* CpuMemoryStack::globalInstance() {
    static CpuMemoryStack* g_inst = nullptr;
    if(g_inst == nullptr) {
        g_inst = new mt_CpuMemoryStackImpl(static_cast<size_t>(1.5 * CLOCKS_PER_SEC));
    }
    return g_inst;
}

CpuMemoryStack* CpuMemoryStack::threadInstance() {
    static boost::thread_specific_ptr<CpuMemoryStack> g_inst;
    if(g_inst.get() == nullptr) {
        g_inst.reset(new CpuMemoryStackImpl(static_cast<size_t>(1.5 * CLOCKS_PER_SEC)));

    }
    return g_inst.get();
}

Allocator* Allocator::getThreadSafeAllocator() {
    static Allocator* g_inst = nullptr;
    if(g_inst == nullptr) {
        g_inst = new mt_UniversalAllocator_t();
    }
    return g_inst;
}

Allocator* Allocator::getThreadSpecificAllocator() {
    if(thread_specific_allocator_unowned)
        return thread_specific_allocator_unowned;
    if(thread_specific_allocator.get() == nullptr) {
        thread_specific_allocator.reset(new mt_UniversalAllocator_t());
    }
    return thread_specific_allocator.get();
}
void Allocator::setThreadSpecificAllocator(Allocator* allocator) {
    cleanupThreadSpecificAllocator();
    thread_specific_allocator_unowned = allocator;
}

void Allocator::cleanupThreadSpecificAllocator() {
    if(auto ptr = thread_specific_allocator.release()) {
        delete ptr;
    }
}

// ================================================================
// CpuStackPolicy
cv::UMatData* CpuStackPolicy::allocate(int dims, const int* sizes, int type,
                                       void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const {
    size_t total = CV_ELEM_SIZE(type);
    for (int i = dims - 1; i >= 0; i--) {
        if (step) {
            if (data && step[i] != CV_AUTOSTEP) {
                CV_Assert(total <= step[i]);
                total = step[i];
            } else {
                step[i] = total;
            }
        }

        total *= static_cast<size_t>(sizes[i]);
    }

    cv::UMatData* u = new cv::UMatData(this);
    u->size = total;

    if (data) {
        u->data = u->origdata = static_cast<uchar*>(data);
        u->flags |= cv::UMatData::USER_ALLOCATED;
    } else {
        void* ptr = 0;
        CpuMemoryStack::threadInstance()->allocate(&ptr, total, CV_ELEM_SIZE(type));

        u->data = u->origdata = static_cast<uchar*>(ptr);
    }

    return u;
}
uchar* CpuStackPolicy::allocate(size_t total) {
    return CpuMemoryStack::threadInstance()->allocate(total);
}

bool CpuStackPolicy::allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const {
    (void)data;
    (void)accessflags;
    (void)usageFlags;
    return false;
}
void CpuStackPolicy::deallocate(uchar* ptr, size_t total) {
    CpuMemoryStack::threadInstance()->deallocate(ptr, total);
}

void CpuStackPolicy::deallocate(cv::UMatData* u) const {
    if (!u)
        return;

    CV_Assert(u->urefcount >= 0);
    CV_Assert(u->refcount >= 0);

    if (u->refcount == 0) {
        if (!(u->flags & cv::UMatData::USER_ALLOCATED)) {
            CpuMemoryStack::threadInstance()->deallocate(u->origdata, u->size);
            u->origdata = 0;
        }

        delete u;
    }
}

// ================================================================
// mt_CpuStackPolicy

cv::UMatData* mt_CpuStackPolicy::allocate(int dims, const int* sizes, int type,
        void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const {
    (void)flags;
    (void)usageFlags;
    size_t total = CV_ELEM_SIZE(type);
    for (int i = dims - 1; i >= 0; i--) {
        if (step) {
            if (data && step[i] != CV_AUTOSTEP) {
                CV_Assert(total <= step[i]);
                total = step[i];
            } else {
                step[i] = total;
            }
        }

        total *= size_t(sizes[i]);
    }

    cv::UMatData* u = new cv::UMatData(this);
    u->size = total;

    if (data) {
        u->data = u->origdata = static_cast<uchar*>(data);
        u->flags |= cv::UMatData::USER_ALLOCATED;
    } else {
        void* ptr = 0;
        CpuMemoryStack::globalInstance()->allocate(&ptr, total, CV_ELEM_SIZE(type));

        u->data = u->origdata = static_cast<uchar*>(ptr);
    }

    return u;
}

bool mt_CpuStackPolicy::allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const {
    (void)data;
    (void)accessflags;
    (void)usageFlags;
    return false;
}

void mt_CpuStackPolicy::deallocate(cv::UMatData* u) const {
    if (!u)
        return;

    CV_Assert(u->urefcount >= 0);
    CV_Assert(u->refcount >= 0);

    if (u->refcount == 0) {
        if (!(u->flags & cv::UMatData::USER_ALLOCATED)) {
            CpuMemoryStack::globalInstance()->deallocate(u->origdata, u->size);
            u->origdata = 0;
        }

        delete u;
    }
}

// ================================================================
// CpuPoolPolicy
cv::UMatData* CpuPoolPolicy::allocate(int dims, const int* sizes, int type,
                                      void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const {
    (void)flags;
    (void)usageFlags;
    size_t total = CV_ELEM_SIZE(type);
    for (int i = dims - 1; i >= 0; i--) {
        if (step) {
            if (data && step[i] != CV_AUTOSTEP) {
                CV_Assert(total <= step[i]);
                total = step[i];
            } else {
                step[i] = total;
            }
        }

        total *= size_t(sizes[i]);
    }

    cv::UMatData* u = new cv::UMatData(this);
    u->size = total;

    if (data) {
        u->data = u->origdata = static_cast<uchar*>(data);
        u->flags |= cv::UMatData::USER_ALLOCATED;
    } else {
        void* ptr = 0;
        CpuMemoryPool::threadInstance()->allocate(&ptr, total, CV_ELEM_SIZE(type));

        u->data = u->origdata = static_cast<uchar*>(ptr);
    }

    return u;
}

bool CpuPoolPolicy::allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const {
    (void)data;
    (void)accessflags;
    (void)usageFlags;
    return false;
}

void CpuPoolPolicy::deallocate(cv::UMatData* u) const {
    if (!u)
        return;

    CV_Assert(u->urefcount >= 0);
    CV_Assert(u->refcount >= 0);

    if (u->refcount == 0) {
        if (!(u->flags & cv::UMatData::USER_ALLOCATED)) {
            CpuMemoryPool::threadInstance()->deallocate(u->origdata, u->size);
            u->origdata = 0;
        }

        delete u;
    }
}
uchar* CpuPoolPolicy::allocate(size_t num_bytes) {
    return CpuMemoryPool::threadInstance()->allocate(num_bytes);
}

void CpuPoolPolicy::deallocate(uchar* ptr, size_t num_bytes) {
    (void)ptr;
    (void)num_bytes;
}

// ================================================================
// mt_CpuPoolPolicy
cv::UMatData* mt_CpuPoolPolicy::allocate(int dims, const int* sizes, int type,
        void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const {
    (void)flags;
    (void)usageFlags;
    size_t total = CV_ELEM_SIZE(type);
    for (int i = dims - 1; i >= 0; i--) {
        if (step) {
            if (data && step[i] != CV_AUTOSTEP) {
                CV_Assert(total <= step[i]);
                total = step[i];
            } else {
                step[i] = total;
            }
        }

        total *= size_t(sizes[i]);
    }

    cv::UMatData* u = new cv::UMatData(this);
    u->size = total;

    if (data) {
        u->data = u->origdata = static_cast<uchar*>(data);
        u->flags |= cv::UMatData::USER_ALLOCATED;
    } else {
        void* ptr = 0;
        CpuMemoryPool::globalInstance()->allocate(&ptr, total, CV_ELEM_SIZE(type));

        u->data = u->origdata = static_cast<uchar*>(ptr);
    }

    return u;
}

bool mt_CpuPoolPolicy::allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const {
    (void)data;
    (void)accessflags;
    (void)usageFlags;
    return false;
}

void mt_CpuPoolPolicy::deallocate(cv::UMatData* u) const {
    if (!u)
        return;

    CV_Assert(u->urefcount >= 0);
    CV_Assert(u->refcount >= 0);

    if (u->refcount == 0) {
        if (!(u->flags & cv::UMatData::USER_ALLOCATED)) {
            CpuMemoryPool::globalInstance()->deallocate(u->origdata, u->size);
            u->origdata = 0;
        }

        delete u;
    }
}


cv::UMatData* PinnedAllocator::allocate(int dims, const int* sizes, int type,
                                        void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const {
    (void)flags;
    (void)usageFlags;
    size_t total = CV_ELEM_SIZE(type);
    for (int i = dims - 1; i >= 0; i--) {
        if (step) {
            if (data && step[i] != CV_AUTOSTEP) {
                CV_Assert(total <= step[i]);
                total = step[i];
            } else {
                step[i] = total;
            }
        }

        total *= size_t(sizes[i]);
    }

    cv::UMatData* u = new cv::UMatData(this);
    u->size = total;

    if (data) {
        u->data = u->origdata = static_cast<uchar*>(data);
        u->flags |= cv::UMatData::USER_ALLOCATED;
    } else {
        void* ptr = 0;
        cudaMallocHost(&ptr, total);

        u->data = u->origdata = static_cast<uchar*>(ptr);
    }

    return u;
}

bool PinnedAllocator::allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const {
    (void)data;
    (void)accessflags;
    (void)usageFlags;
    return false;
}

void PinnedAllocator::deallocate(cv::UMatData* u) const {
    if (!u)
        return;

    CV_Assert(u->urefcount >= 0);
    CV_Assert(u->refcount >= 0);

    if (u->refcount == 0) {
        if (!(u->flags & cv::UMatData::USER_ALLOCATED)) {
            cudaFreeHost(u->origdata);
            u->origdata = 0;
        }

        delete u;
    }
}
