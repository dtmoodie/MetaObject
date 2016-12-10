#pragma once

#include "Export.hpp"
#include "MemoryBlock.h"
#include <opencv2/core/cuda.hpp>
#include <boost/thread/mutex.hpp>
#include <list>
namespace mo
{
MO_EXPORTS inline unsigned char* alignMemory(unsigned char* ptr, int elemSize);
MO_EXPORTS inline int alignmentOffset(unsigned char* ptr, int elemSize);
MO_EXPORTS void SetScopeName(const std::string& name);
MO_EXPORTS const std::string& GetScopeName();

/// ========================================================
/// Memory layout policies
/// ========================================================

/*!
 * \brief The PitchedPolicy class allocates memory with padding
 *        such that a 2d array can be utilized as a texture reference
 */
class MO_EXPORTS PitchedPolicy
{
public:
    inline PitchedPolicy();
    inline void SizeNeeded(int rows, int cols, int elemSize,
                           size_t& sizeNeeded, size_t& stride);
private:
    size_t textureAlignment;
};

/*!
 * \brief The ContinuousPolicy class allocates memory with zero padding
 *        wich allows for nice reshaping operations
 *
 */
class MO_EXPORTS ContinuousPolicy
{
public:
    inline void SizeNeeded(int rows, int cols, int elemSize,
                           size_t& sizeNeeded, size_t& stride);
};

/// ========================================================
/// Allocation Policies
/// ========================================================

/*!
 * \brief The AllocationPolicy class is a base for all other allocation
 *        policies, it's members track memory usage by the allocator
 */
class MO_EXPORTS AllocationPolicy
{
public:
    /*!
     * \brief GetMemoryUsage
     * \return current estimated memory usage
     */
    inline size_t GetMemoryUsage() const;
    virtual void Release() {}
protected:
    size_t memoryUsage;
    /*!
     * \brief current_allocations keeps track of the stacks allocated by allocate(size_t)
     *        since a call to free(unsigned char*) will not return the size of the allocated
     *        data
     */
    std::map<unsigned char*, size_t> current_allocations;
};

/*!
 * \brief The PoolPolicy allocation policy uses a memory pool to cache
 *        memory usage of small amounts of data.  Best used for variable
 *        amounts of data
 */
template<typename T, typename PaddingPolicy>
class MO_EXPORTS PoolPolicy
{

};

/// ========================================================================================
template<typename PaddingPolicy>
class MO_EXPORTS PoolPolicy<cv::cuda::GpuMat, PaddingPolicy>
        : public virtual AllocationPolicy
        , public virtual PaddingPolicy
{
public:
    PoolPolicy(size_t initialBlockSize = 1e7);

    inline bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
    inline void free(cv::cuda::GpuMat* mat);

    inline unsigned char* allocate(size_t num_bytes);
    inline void free(unsigned char* ptr);
    virtual void Release();
private:
    size_t _initial_block_size;
    std::list<std::shared_ptr<GpuMemoryBlock>> blocks;
};


class MO_EXPORTS CpuPoolPolicy: virtual public cv::MatAllocator
{
public:
    cv::UMatData* allocate(int dims, const int* sizes, int type,
        void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const;
    bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const;
    void deallocate(cv::UMatData* data) const;
    void Release() {}
};

class MO_EXPORTS mt_CpuPoolPolicy : virtual public CpuPoolPolicy
{
public:
    cv::UMatData* allocate(int dims, const int* sizes, int type,
        void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const;
    bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const;
    void deallocate(cv::UMatData* data) const;
};
class MO_EXPORTS PinnedAllocator : virtual public cv::MatAllocator
{
public:
    cv::UMatData* allocate(int dims, const int* sizes, int type,
        void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const;
    bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const;
    void deallocate(cv::UMatData* data) const;
};


/*!
 *  \brief The StackPolicy class checks for a free memory stack of the exact
 *         requested size, if it is available it allocates from the free stack.
 *         If it wasn't able to find memory of the exact requested size, it will
 *         allocate the exact size and return it.  Since memory is not coelesced
 *         between stacks, this is best for large fixed size data, such as
 *         repeatedly allocated and deallocated images.
 */
template<typename T, typename PaddingPolicy> class MO_EXPORTS StackPolicy{};

/// =================================================================================
template<typename PaddingPolicy>
class MO_EXPORTS StackPolicy<cv::cuda::GpuMat, PaddingPolicy>
        : public virtual AllocationPolicy
        , public virtual PaddingPolicy
{
public:
    typedef cv::cuda::GpuMat MatType;
    StackPolicy();
    bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
    void free(cv::cuda::GpuMat* mat);

    unsigned char* allocate(size_t num_bytes);
    void free(unsigned char* ptr);
    virtual void Release();
protected:
    void clear();
    struct FreeMemory
    {
        FreeMemory(unsigned char* ptr_, clock_t time_, size_t size_):
            ptr(ptr_), free_time(time_), size(size_){}
        unsigned char* ptr;
        clock_t free_time;
        size_t size;
    };
    std::list<FreeMemory> deallocateList;
    size_t deallocateDelay; // ms
};


class MO_EXPORTS CpuStackPolicy
    : public virtual AllocationPolicy
    , public virtual ContinuousPolicy
    , public virtual cv::MatAllocator
{
public:
    typedef cv::Mat MatType;
    cv::UMatData* allocate(int dims, const int* sizes, int type,
        void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const;
    bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const;
    void deallocate(cv::UMatData* data) const;
    void Release(){}
};

class MO_EXPORTS mt_CpuStackPolicy: virtual public CpuStackPolicy
{
public:
    typedef cv::Mat MatType;
    cv::UMatData* allocate(int dims, const int* sizes, int type,
        void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const;
    bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const;
    void deallocate(cv::UMatData* data) const;
};


/*!
 *  \brief The NonCachingPolicy allocates and deallocates the same as
 *         OpenCV's default allocator.  It has the advantage of memory
 *         usage tracking.
 */

template<typename T, typename PaddingPolicy> class MO_EXPORTS NonCachingPolicy {};

template<typename PaddingPolicy>
class MO_EXPORTS NonCachingPolicy<cv::cuda::GpuMat, PaddingPolicy>
        : public virtual AllocationPolicy
{
public:
    bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
    void free(cv::cuda::GpuMat* mat);
    unsigned char* allocate(size_t num_bytes);
    void free(unsigned char* ptr);
};

template<class Allocator, class MatType>
class LockPolicyImpl: public Allocator{};

template<class Allocator>
class LockPolicyImpl<Allocator, cv::cuda::GpuMat>: public Allocator
{
public:
    typedef cv::cuda::GpuMat MatType;
    inline bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
    inline void free(cv::cuda::GpuMat* mat);

    inline unsigned char* allocate(size_t num_bytes);
    inline void free(unsigned char* ptr);
private:
    boost::mutex mtx;
};

template<class Allocator>
class LockPolicyImpl<Allocator, cv::Mat>: public Allocator
{
public:
    typedef cv::Mat MatType;
    cv::UMatData* allocate(int dims, const int* sizes, int type,
        void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const;
    bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const;
    void deallocate(cv::UMatData* data) const;
private:
    boost::mutex mtx;
};

class MO_EXPORTS CpuMemoryPool
{
public:
    virtual ~CpuMemoryPool() {}
    static CpuMemoryPool* GlobalInstance();
    static CpuMemoryPool* ThreadInstance();
    virtual bool allocate(void** ptr, size_t total, size_t elemSize) = 0;
    virtual bool deallocate(void* ptr, size_t total) = 0;
};

class MO_EXPORTS CpuMemoryStack
{
public:
    virtual ~CpuMemoryStack() {}
    static CpuMemoryStack* GlobalInstance();
    static CpuMemoryStack* ThreadInstance();
    virtual bool allocate(void** ptr, size_t total, size_t elemSize) = 0;
    virtual bool deallocate(void* ptr, size_t total) = 0;
};


/*!
 * \brief The LockPolicy class locks calls to the given allocator
 */
template<class Allocator>
class MO_EXPORTS LockPolicy
        : public LockPolicyImpl<Allocator, typename Allocator::MatType>
{

};

template<class SmallAllocator, class LargeAllocator, class MatType>
class MO_EXPORTS CombinedPolicyImpl
        : virtual public SmallAllocator
        , virtual public LargeAllocator
{

};

template<class SmallAllocator, class LargeAllocator>
class MO_EXPORTS CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::cuda::GpuMat>
        : virtual public SmallAllocator
        , virtual public LargeAllocator
{
public:
    CombinedPolicyImpl(size_t threshold);
    inline bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
    inline void free(cv::cuda::GpuMat* mat);

    inline unsigned char* allocate(size_t num_bytes);
    inline void free(unsigned char* ptr);
    void Release();
private:
    size_t threshold;
};

template<class SmallAllocator, class LargeAllocator>
class MO_EXPORTS CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::Mat>
        : virtual public SmallAllocator
        , virtual public LargeAllocator
{
public:
    CombinedPolicyImpl(size_t threshold);
    inline cv::UMatData* allocate(int dims, const int* sizes, int type,
                                  void* data, size_t* step, int flags,
                                  cv::UMatUsageFlags usageFlags) const;
    inline bool allocate(cv::UMatData* data, int accessflags,
                         cv::UMatUsageFlags usageFlags) const;
    inline void deallocate(cv::UMatData* data) const;
    void Release();
private:
    size_t threshold;
};

template<class SmallAllocator, class LargeAllocator>
class MO_EXPORTS CombinedPolicy
        : public CombinedPolicyImpl<SmallAllocator, LargeAllocator, typename LargeAllocator::MatType>
{
public:
    typedef typename LargeAllocator::MatType MatType;
    CombinedPolicy(size_t threshold = 1e7);
};

/*!
 * \brief The ScopeDebugPolicy class
 */
template<class Allocator, class MatType>
class MO_EXPORTS ScopeDebugPolicy: public virtual Allocator{};

template<class Allocator>
class MO_EXPORTS ScopeDebugPolicy<Allocator, cv::cuda::GpuMat>: public virtual Allocator
{
public:
    inline bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
    inline void free(cv::cuda::GpuMat* mat);

    inline unsigned char* allocate(size_t num_bytes);
    inline void free(unsigned char* ptr);
private:
    std::map<unsigned char*, std::string> scopeOwnership;
    std::map<std::string, size_t> scopedAllocationSize;
};

class MO_EXPORTS Allocator:
        virtual public cv::cuda::GpuMat::Allocator,
        virtual public cv::MatAllocator
{
public:
    static Allocator* GetThreadSafeAllocator();
    static Allocator* GetThreadSpecificAllocator();
    virtual void Release() {}
};


}
