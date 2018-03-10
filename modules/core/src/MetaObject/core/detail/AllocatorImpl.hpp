#pragma once
#include "Allocator.hpp"
#include "MetaObject/logging/logging.hpp"
#include <MetaObject/thread/cuda.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/tss.hpp>
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudev/common.hpp>

#define MO_CUDA_ERROR_CHECK(expr, msg)                                                                                 \
    {                                                                                                                  \
        cudaError_t err = (expr);                                                                                      \
        if (err != cudaSuccess)                                                                                        \
        {                                                                                                              \
            THROW(warning) << #expr << " failed " << cudaGetErrorString(err) << " " msg;                               \
        }                                                                                                              \
    }

namespace mo
{

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
        inline void sizeNeeded(int rows, int cols, int elemSize, size_t& size_needed, size_t& stride);

      private:
        size_t texture_alignment;
    };

    /*!
    * \brief The ContinuousPolicy class allocates memory with zero padding
    *        wich allows for nice reshaping operations
    *
    */
    class MO_EXPORTS ContinuousPolicy
    {
      public:
        inline void sizeNeeded(int rows, int cols, int elemSize, size_t& size_needed, size_t& stride);
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
        virtual ~AllocationPolicy();
        /*!
        * \brief GetMemoryUsage
        * \return current estimated memory usage
        */
        inline size_t getMemoryUsage() const { return m_memory_usage; }
        virtual void release() {}
      protected:
        size_t m_memory_usage = 0;
        /*!
        * \brief current_allocations keeps track of the stacks allocated by allocate(size_t)
        *        since a call to free(unsigned char*) will not return the size of the allocated
        *        data
        */
        std::map<unsigned char*, size_t> m_current_allocations;
    };

    /*!
    * \brief The PoolPolicy allocation policy uses a memory pool to cache
    *        memory usage of small amounts of data.  Best used for variable
    *        amounts of data
    */
    template <typename T, typename PaddingPolicy>
    class MO_EXPORTS PoolPolicy
    {
    };

    /// ========================================================================================
    template <typename PaddingPolicy>
    class MO_EXPORTS PoolPolicy<cv::cuda::GpuMat, PaddingPolicy> : public virtual AllocationPolicy,
                                                                   public virtual PaddingPolicy
    {
      public:
        typedef cv::cuda::GpuMat MatType;
        PoolPolicy(size_t initial_block_size = 1e7);
        PoolPolicy(const PoolPolicy&) = delete;
        PoolPolicy& operator=(const PoolPolicy&) = delete;

        inline bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
        inline void free(cv::cuda::GpuMat* mat);

        inline unsigned char* allocate(size_t num_bytes);
        inline void deallocate(unsigned char* ptr, size_t num_bytes);
        virtual void release();

      private:
        size_t m_initial_block_size;
        std::list<std::unique_ptr<GpuMemoryBlock>> m_blocks;
    };

    class MO_EXPORTS CpuPoolPolicy : virtual public cv::MatAllocator
    {
      public:
        cv::UMatData* allocate(int dims,
                               const int* sizes,
                               int type,
                               void* data,
                               size_t* step,
                               int flags,
                               cv::UMatUsageFlags usage_flags) const;
        bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usage_flags) const;
        void deallocate(cv::UMatData* data) const;
        uchar* allocate(size_t num_bytes);
        void deallocate(uchar* ptr, size_t num_bytes);
        void release() {}
    };

    class MO_EXPORTS mt_CpuPoolPolicy : virtual public CpuPoolPolicy
    {
      public:
        cv::UMatData* allocate(int dims,
                               const int* sizes,
                               int type,
                               void* data,
                               size_t* step,
                               int flags,
                               cv::UMatUsageFlags usage_flags) const;
        bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usage_flags) const;
        void deallocate(cv::UMatData* data) const;
        uchar* allocate(size_t num_bytes);
        void deallocate(uchar* ptr, size_t num_bytes);
    };
    class MO_EXPORTS PinnedAllocator : virtual public cv::MatAllocator
    {
      public:
        cv::UMatData* allocate(int dims,
                               const int* sizes,
                               int type,
                               void* data,
                               size_t* step,
                               int flags,
                               cv::UMatUsageFlags usageFlags) const;
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
    template <typename T, typename PaddingPolicy>
    class MO_EXPORTS StackPolicy
    {
    };

    /// =================================================================================
    template <typename PaddingPolicy>
    class MO_EXPORTS StackPolicy<cv::cuda::GpuMat, PaddingPolicy> : public virtual AllocationPolicy,
                                                                    public virtual PaddingPolicy
    {
      public:
        typedef cv::cuda::GpuMat MatType;
        StackPolicy();

        bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
        void free(cv::cuda::GpuMat* mat);

        unsigned char* allocate(size_t num_bytes);
        void deallocate(unsigned char* ptr, size_t num_bytes);

        virtual void release();

      protected:
        void clear();
        struct FreeMemory
        {
            FreeMemory(unsigned char* ptr_, clock_t time_, size_t size_) : ptr(ptr_), free_time(time_), size(size_) {}
            unsigned char* ptr;
            clock_t free_time;
            size_t size;
        };
        std::list<FreeMemory> m_deallocate_list;
        size_t m_deallocate_delay; // ms
    };

    class MO_EXPORTS CpuStackPolicy : public virtual AllocationPolicy,
                                      public virtual ContinuousPolicy,
                                      public virtual cv::MatAllocator
    {
      public:
        typedef cv::Mat MatType;

        cv::UMatData* allocate(int dims,
                               const int* sizes,
                               int type,
                               void* data,
                               size_t* step,
                               int flags,
                               cv::UMatUsageFlags usageFlags) const;
        bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const;
        void deallocate(cv::UMatData* data) const;

        uchar* allocate(size_t total);
        void deallocate(uchar* ptr, size_t total);
        void release() {}
    };

    class MO_EXPORTS mt_CpuStackPolicy : virtual public CpuStackPolicy
    {
      public:
        typedef cv::Mat MatType;

        cv::UMatData* allocate(int dims,
                               const int* sizes,
                               int type,
                               void* data,
                               size_t* step,
                               int flags,
                               cv::UMatUsageFlags usageFlags) const;
        bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const;
        void deallocate(cv::UMatData* data) const;

        uchar* allocate(size_t total);
        bool deallocate(uchar* ptr, size_t total);
    };

    /*!
    *  \brief The NonCachingPolicy allocates and deallocates the same as
    *         OpenCV's default allocator.  It has the advantage of memory
    *         usage tracking.
    */

    template <typename T, typename PaddingPolicy>
    class MO_EXPORTS NonCachingPolicy
    {
    };

    template <typename PaddingPolicy>
    class MO_EXPORTS NonCachingPolicy<cv::cuda::GpuMat, PaddingPolicy> : public virtual AllocationPolicy
    {
      public:
        bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
        void free(cv::cuda::GpuMat* mat);
        unsigned char* allocate(size_t num_bytes);
        void deallocate(unsigned char* ptr, size_t num_bytes);
    };

    template <class Allocator, class MatType>
    class LockPolicyImpl : public Allocator
    {
    };

    template <class Allocator>
    class LockPolicyImpl<Allocator, cv::cuda::GpuMat> : public Allocator
    {
      public:
        typedef cv::cuda::GpuMat MatType;
        inline bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
        inline void free(cv::cuda::GpuMat* mat);

        inline unsigned char* allocate(size_t num_bytes);
        inline void deallocate(unsigned char* ptr, size_t num_bytes);

      private:
        boost::mutex m_mtx;
    };

    template <class Allocator>
    class LockPolicyImpl<Allocator, cv::Mat> : public Allocator
    {
      public:
        typedef cv::Mat MatType;
        cv::UMatData* allocate(int dims,
                               const int* sizes,
                               int type,
                               void* data,
                               size_t* step,
                               int flags,
                               cv::UMatUsageFlags usageFlags) const;
        bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const;
        void deallocate(cv::UMatData* data) const;
        inline unsigned char* allocate(size_t num_bytes);
        inline void deallocate(unsigned char* ptr, size_t num_bytes);

      private:
        boost::mutex m_mtx;
    };

    template <class Allocator, class MatType>
    class RefCountPolicyImpl
    {
    };

    template <class Allocator>
    class RefCountPolicyImpl<Allocator, cv::Mat> : public Allocator
    {
      public:
        typedef cv::Mat MatType;

        template <class... T>
        RefCountPolicyImpl(T... args) : Allocator(args...)
        {
        }

        ~RefCountPolicyImpl();
        cv::UMatData* allocate(int dims,
                               const int* sizes,
                               int type,
                               void* data,
                               size_t* step,
                               int flags,
                               cv::UMatUsageFlags usageFlags) const;
        bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const;
        void deallocate(cv::UMatData* data) const;
        inline unsigned char* allocate(size_t num_bytes);
        inline void deallocate(unsigned char* ptr, size_t num_bytes);

      private:
        int m_ref_count = 0;
    };

    template <class Allocator>
    class RefCountPolicyImpl<Allocator, cv::cuda::GpuMat> : public Allocator
    {
      public:
        typedef cv::cuda::GpuMat MatType;

        template <class... T>
        RefCountPolicyImpl(T... args) : Allocator(args...)
        {
        }
        ~RefCountPolicyImpl();
        inline bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
        inline void free(cv::cuda::GpuMat* mat);

        inline unsigned char* allocate(size_t num_bytes);
        inline void deallocate(unsigned char* ptr, size_t num_bytes);

      private:
        int m_ref_count = 0;
    };

    class MO_EXPORTS CpuMemoryPool
    {
      public:
        virtual ~CpuMemoryPool() {}
        static CpuMemoryPool* globalInstance();
        static CpuMemoryPool* threadInstance();
        virtual bool allocate(void** ptr, size_t total, size_t elemSize) = 0;
        virtual uchar* allocate(size_t total) = 0;
        virtual bool deallocate(void* ptr, size_t total) = 0;
    };

    class MO_EXPORTS CpuMemoryStack
    {
      public:
        virtual ~CpuMemoryStack() {}
        static CpuMemoryStack* globalInstance();
        static CpuMemoryStack* threadInstance();
        virtual bool allocate(void** ptr, size_t total, size_t elemSize) = 0;
        virtual uchar* allocate(size_t total) = 0;
        virtual bool deallocate(void* ptr, size_t total) = 0;
    };

    /*!
    * \brief The LockPolicy class locks calls to the given allocator
    */
    template <class Allocator>
    class MO_EXPORTS LockPolicy : public LockPolicyImpl<Allocator, typename Allocator::MatType>
    {
    };

    /*!
    *  \brief The ref count policy keeps a count of the number of mats that have been
    *         allocated and deallocated so that you can debug when deleting an allocator
    *         prior to releasing all allocated mats
    */
    template <class Allocator>
    class MO_EXPORTS RefCountPolicy : public RefCountPolicyImpl<Allocator, typename Allocator::MatType>
    {
      public:
        template <class... T>
        RefCountPolicy(T... args) : RefCountPolicyImpl<Allocator, typename Allocator::MatType>(args...)
        {
        }
    };

    template <class SmallAllocator, class LargeAllocator, class MatType>
    class MO_EXPORTS CombinedPolicyImpl : virtual public SmallAllocator, virtual public LargeAllocator
    {
    };

    template <class SmallAllocator, class LargeAllocator>
    class MO_EXPORTS CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::cuda::GpuMat>
        : virtual public SmallAllocator, virtual public LargeAllocator
    {
      public:
        CombinedPolicyImpl(size_t threshold);
        inline bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
        inline void free(cv::cuda::GpuMat* mat);

        inline unsigned char* allocate(size_t num_bytes);
        inline void deallocate(unsigned char* ptr, size_t num_bytes);
        void release();

      private:
        size_t threshold;
    };

    template <class SmallAllocator, class LargeAllocator>
    class MO_EXPORTS CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::Mat> : virtual public SmallAllocator,
                                                                                   virtual public LargeAllocator
    {
      public:
        CombinedPolicyImpl(size_t threshold);
        inline cv::UMatData* allocate(int dims,
                                      const int* sizes,
                                      int type,
                                      void* data,
                                      size_t* step,
                                      int flags,
                                      cv::UMatUsageFlags usageFlags) const;
        inline bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const;
        inline void deallocate(cv::UMatData* data) const;
        inline unsigned char* allocate(size_t num_bytes);
        inline void deallocate(unsigned char* ptr, size_t num_bytes);
        void release();

      private:
        size_t threshold;
    };

    template <class SmallAllocator, class LargeAllocator>
    class MO_EXPORTS CombinedPolicy
        : public CombinedPolicyImpl<SmallAllocator, LargeAllocator, typename LargeAllocator::MatType>
    {
      public:
        typedef typename LargeAllocator::MatType MatType;
        CombinedPolicy(size_t threshold = 1 * 1024 * 512);
    };

    /*!
    * \brief The ScopeDebugPolicy class
    */
    template <class Allocator, class MatType>
    class MO_EXPORTS ScopeDebugPolicy : public virtual Allocator
    {
    };

    template <class Allocator>
    class MO_EXPORTS ScopeDebugPolicy<Allocator, cv::cuda::GpuMat> : public virtual Allocator
    {
      public:
        inline bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
        inline void free(cv::cuda::GpuMat* mat);

        inline unsigned char* allocate(size_t num_bytes);
        inline void deallocate(unsigned char* ptr, size_t numBytes);

      private:
        std::map<unsigned char*, std::string> scopeOwnership;
        std::map<std::string, size_t> scopedAllocationSize;
    };

    unsigned char* alignMemory(unsigned char* ptr, int elem_size) { return ptr + alignmentOffset(ptr, elem_size); }

    const unsigned char* alignMemory(const unsigned char* ptr, int elem_size)
    {
        return ptr + alignmentOffset(ptr, elem_size);
    }

    size_t alignmentOffset(const unsigned char* ptr, size_t elem_size)
    {
        return elem_size - (reinterpret_cast<const size_t>(ptr) % elem_size);
    }

    /// ==========================================================
    /// PitchedPolicy
    PitchedPolicy::PitchedPolicy()
    {
        texture_alignment = cv::cuda::DeviceInfo(cv::cuda::getDevice()).textureAlignment();
    }

    void PitchedPolicy::sizeNeeded(int rows, int cols, int elemSize, size_t& size_needed, size_t& stride)
    {
        if (rows == 1 || cols == 1)
        {
            stride = static_cast<size_t>(cols * elemSize);
        }
        else
        {
            if ((static_cast<size_t>(cols * elemSize) % texture_alignment) == 0)
            {
                stride = static_cast<size_t>(cols * elemSize);
            }
            else
            {
                stride = static_cast<size_t>(cols * elemSize) + texture_alignment -
                         (static_cast<size_t>(cols * elemSize) % texture_alignment);
            }
        }
        size_needed = stride * static_cast<size_t>(rows);
    }

    /// ==========================================================
    /// ContinuousPolicy
    void ContinuousPolicy::sizeNeeded(int rows, int cols, int elemSize, size_t& size_needed, size_t& stride)
    {
        stride = static_cast<size_t>(cols * elemSize);
        size_needed = stride * static_cast<size_t>(rows);
    }

    /// ==========================================================
    /// PoolPolicy
    template <typename PaddingPolicy>
    PoolPolicy<cv::cuda::GpuMat, PaddingPolicy>::PoolPolicy(size_t initial_block_size)
        : m_initial_block_size(initial_block_size)
    {
        m_blocks.emplace_back(std::unique_ptr<GpuMemoryBlock>(new GpuMemoryBlock(initial_block_size)));
    }

    template <typename PaddingPolicy>
    bool
    PoolPolicy<cv::cuda::GpuMat, PaddingPolicy>::allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elem_size)
    {
        size_t size_needed, stride;
        PaddingPolicy::sizeNeeded(rows, cols, static_cast<int>(elem_size), size_needed, stride);
        unsigned char* ptr;
        for (auto& itr : m_blocks)
        {
            ptr = itr->allocate(size_needed, elem_size);
            if (ptr)
            {
                mat->data = ptr;
                mat->step = stride;
                mat->refcount = static_cast<int*>(cv::fastMalloc(sizeof(int)));
                m_memory_usage += mat->step * rows;
                return true;
            }
        }
        // If we get to this point, then no memory was found, need to allocate new memory
        m_blocks.push_back(std::unique_ptr<GpuMemoryBlock>(new GpuMemoryBlock(std::max(m_initial_block_size / 2, size_needed))));
        MO_LOG(trace) << "[GPU] Expanding memory pool by "
                      << std::max(m_initial_block_size / 2, size_needed) / (1024 * 1024) << " MB";
        if (unsigned char* ptr = (*m_blocks.rbegin())->allocate(size_needed, elem_size))
        {
            mat->data = ptr;
            mat->step = stride;
            mat->refcount = static_cast<int*>(cv::fastMalloc(sizeof(int)));
            m_memory_usage += mat->step * rows;
            return true;
        }
        return false;
    }

    template <typename PaddingPolicy>
    void PoolPolicy<cv::cuda::GpuMat, PaddingPolicy>::free(cv::cuda::GpuMat* mat)
    {
        for (auto& itr : m_blocks)
        {
            if (itr->deAllocate(mat->data))
            {
                cv::fastFree(mat->refcount);
                m_memory_usage -= mat->step * mat->rows;
                return;
            }
        }
        throw cv::Exception(0, "[GPU] Unable to find memory to deallocate", __FUNCTION__, __FILE__, __LINE__);
    }

    template <typename PaddingPolicy>
    unsigned char* PoolPolicy<cv::cuda::GpuMat, PaddingPolicy>::allocate(size_t size_needed)
    {
        unsigned char* ptr;
        for (auto& itr : m_blocks)
        {
            ptr = itr->allocate(size_needed, 1);
            if (ptr)
            {
                m_memory_usage += size_needed;
                return ptr;
            }
        }
        // If we get to this point, then no memory was found, need to allocate new memory
        m_blocks.push_back(std::unique_ptr<GpuMemoryBlock>(new GpuMemoryBlock(std::max(m_initial_block_size / 2, size_needed))));
        if (unsigned char* ptr = (*m_blocks.rbegin())->allocate(size_needed, 1))
        {
            m_memory_usage += size_needed;
            return ptr;
        }
        return nullptr;
    }

    template <typename PaddingPolicy>
    void PoolPolicy<cv::cuda::GpuMat, PaddingPolicy>::deallocate(unsigned char* ptr, size_t /*num_bytes*/)
    {
        for (auto& itr : m_blocks)
        {
            if (itr->deAllocate(ptr))
            {
                return;
            }
        }
    }

    template <typename PaddingPolicy>
    void PoolPolicy<cv::cuda::GpuMat, PaddingPolicy>::release()
    {
        m_blocks.clear();
    }

    /// ==========================================================
    /// StackPolicy
    template <typename PaddingPolicy>
    StackPolicy<cv::cuda::GpuMat, PaddingPolicy>::StackPolicy()
    {
#ifdef _MSC_VER
        m_deallocate_delay = 1000;
#else
        m_deallocate_delay = 1000 * 1000;
#endif
    }

    template <typename PaddingPolicy>
    bool
    StackPolicy<cv::cuda::GpuMat, PaddingPolicy>::allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
    {
        size_t size_needed, stride;

        PaddingPolicy::sizeNeeded(rows, cols, static_cast<int>(elemSize), size_needed, stride);
        typedef typename std::list<typename StackPolicy<cv::cuda::GpuMat, PaddingPolicy>::FreeMemory>::iterator Itr;
        std::vector<std::pair<clock_t, Itr>> candidates;
        clock_t time = clock();
        for (auto itr = m_deallocate_list.begin(); itr != m_deallocate_list.end(); ++itr)
        {
            if (itr->size == size_needed)
            {
                candidates.emplace_back((time - itr->free_time), itr);
            }
        }
        if (candidates.size())
        {
            auto best_candidate =
                std::max_element(candidates.begin(),
                                 candidates.end(),
                                 [](const std::pair<clock_t, Itr>& i1, const std::pair<clock_t, Itr>& i2) {
                                     return i1.first < i2.first;
                                 });
            mat->data = best_candidate->second->ptr;
            mat->step = stride;
            mat->refcount = static_cast<int*>(cv::fastMalloc(sizeof(int)));
            m_memory_usage += size_needed;
            m_deallocate_list.erase(best_candidate->second);
            return true;
        }
        if (rows > 1 && cols > 1)
        {
            MO_CUDA_ERROR_CHECK(
                cudaMallocPitch(
                    &mat->data, &mat->step, elemSize * static_cast<size_t>(cols), static_cast<size_t>(rows)),
                << " while allocating " << rows << ", " << cols << ". Elemsize: " << elemSize << " Total allocation: "
                << elemSize * cols * rows / (1024 * 1024)
                << " MB");

            m_memory_usage += mat->step * rows;
        }
        else
        {
            CV_CUDEV_SAFE_CALL(cudaMalloc(&mat->data, elemSize * cols * rows));
            m_memory_usage += mat->step * rows;
            mat->step = elemSize * static_cast<size_t>(cols);
        }
        mat->refcount = static_cast<int*>(cv::fastMalloc(sizeof(int)));
        return true;
    }

    template <typename PaddingPolicy>
    void StackPolicy<cv::cuda::GpuMat, PaddingPolicy>::free(cv::cuda::GpuMat* mat)
    {
        m_deallocate_list.emplace_back(mat->datastart, clock(), mat->dataend - mat->datastart);
        cv::fastFree(mat->refcount);
        clear();
    }

    template <typename PaddingPolicy>
    unsigned char* StackPolicy<cv::cuda::GpuMat, PaddingPolicy>::allocate(size_t size_needed)
    {
        unsigned char* ptr = nullptr;
        for (auto itr = m_deallocate_list.begin(); itr != m_deallocate_list.end(); ++itr)
        {
            if (itr->size == size_needed)
            {
                ptr = itr->ptr;
                this->m_memory_usage += size_needed;
                m_current_allocations[ptr] = size_needed;
                m_deallocate_list.erase(itr);
                return ptr;
            }
        }
        CV_CUDEV_SAFE_CALL(cudaMalloc(&ptr, size_needed));
        this->m_memory_usage += size_needed;
        m_current_allocations[ptr] = size_needed;
        return ptr;
    }

    template <typename PaddingPolicy>
    void StackPolicy<cv::cuda::GpuMat, PaddingPolicy>::deallocate(unsigned char* ptr, size_t /*num_bytes*/)
    {
        auto itr = m_current_allocations.find(ptr);
        if (itr != m_current_allocations.end())
        {
            m_current_allocations.erase(itr);
            m_deallocate_list.emplace_back(ptr, clock(), m_current_allocations[ptr]);
        }

        clear();
    }

    template <typename PaddingPolicy>
    void StackPolicy<cv::cuda::GpuMat, PaddingPolicy>::clear()
    {
        if (isCudaThread())
            return;
        auto time = clock();
        for (auto itr = m_deallocate_list.begin(); itr != m_deallocate_list.end();)
        {
            if ((time - itr->free_time) > m_deallocate_delay)
            {
                m_memory_usage -= itr->size;
                MO_LOG(trace) << "[GPU] Deallocating block of size " << itr->size / (1024 * 1024)
                              << "MB. Which was stale for " << (time - itr->free_time) * 1000 / CLOCKS_PER_SEC
                              << " ms at " << static_cast<void*>(itr->ptr);
                CV_CUDEV_SAFE_CALL(cudaFree(itr->ptr));
                itr = m_deallocate_list.erase(itr);
            }
            else
            {
                ++itr;
            }
        }
    }

    template <typename PaddingPolicy>
    void StackPolicy<cv::cuda::GpuMat, PaddingPolicy>::release()
    {
        if (isCudaThread())
            return;
        for (auto& itr : m_deallocate_list)
        {
            CV_CUDEV_SAFE_CALL(cudaFree(itr.ptr));
        }
        m_deallocate_list.clear();
    }

    /// ==========================================================
    /// NonCachingPolicy
    template <typename PaddingPolicy>
    bool NonCachingPolicy<cv::cuda::GpuMat, PaddingPolicy>::allocate(cv::cuda::GpuMat* mat,
                                                                     int rows,
                                                                     int cols,
                                                                     size_t elem_size)
    {
        size_t size_needed, stride;
        PaddingPolicy::sizeNeeded(rows, cols, elem_size, size_needed, stride);
        if (rows > 1 && cols > 1)
        {
            CV_CUDEV_SAFE_CALL(cudaMallocPitch(&mat->data, &mat->step, elem_size * cols, rows));
            m_memory_usage += mat->step * rows;
        }
        else
        {
            CV_CUDEV_SAFE_CALL(cudaMalloc(&mat->data, elem_size * cols * rows));
            m_memory_usage += elem_size * cols * rows;
            mat->step = elem_size * static_cast<size_t>(cols);
        }
        mat->refcount = static_cast<int*>(cv::fastMalloc(sizeof(int)));
        return true;
    }

    template <typename PaddingPolicy>
    void NonCachingPolicy<cv::cuda::GpuMat, PaddingPolicy>::free(cv::cuda::GpuMat* mat)
    {
        if (isCudaThread())
            return;
        CV_CUDEV_SAFE_CALL(cudaFree(mat->data));
        cv::fastFree(mat->refcount);
    }

    template <typename PaddingPolicy>
    unsigned char* NonCachingPolicy<cv::cuda::GpuMat, PaddingPolicy>::allocate(size_t num_bytes)
    {
        unsigned char* ptr = nullptr;
        CV_CUDEV_SAFE_CALL(cudaMalloc(&ptr, num_bytes));
        m_memory_usage += num_bytes;
        return ptr;
    }

    template <typename PaddingPolicy>
    void NonCachingPolicy<cv::cuda::GpuMat, PaddingPolicy>::deallocate(unsigned char* ptr, size_t /*num_bytes*/)
    {
        if (isCudaThread())
            return;
        CV_CUDEV_SAFE_CALL(cudaFree(ptr));
    }

    /// ==========================================================
    /// LockPolicy
    template <class Allocator>
    bool
    LockPolicyImpl<Allocator, cv::cuda::GpuMat>::allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
    {
        boost::mutex::scoped_lock lock(m_mtx);
        return Allocator::allocate(mat, rows, cols, elemSize);
    }

    template <class Allocator>
    void LockPolicyImpl<Allocator, cv::cuda::GpuMat>::free(cv::cuda::GpuMat* mat)
    {
        boost::mutex::scoped_lock lock(m_mtx);
        return Allocator::free(mat);
    }

    template <class Allocator>
    unsigned char* LockPolicyImpl<Allocator, cv::cuda::GpuMat>::allocate(size_t num_bytes)
    {
        boost::mutex::scoped_lock lock(m_mtx);
        return Allocator::allocate(num_bytes);
    }

    template <class Allocator>
    void LockPolicyImpl<Allocator, cv::cuda::GpuMat>::deallocate(unsigned char* ptr, size_t num_bytes)
    {
        boost::mutex::scoped_lock lock(m_mtx);
        return Allocator::deallocate(ptr, num_bytes);
    }

    template <class Allocator>
    cv::UMatData* LockPolicyImpl<Allocator, cv::Mat>::allocate(
        int dims, const int* sizes, int type, void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const
    {
        return Allocator::allocate(dims, sizes, type, data, step, flags, usageFlags);
    }

    template <class Allocator>
    bool LockPolicyImpl<Allocator, cv::Mat>::allocate(cv::UMatData* data,
                                                      int accessflags,
                                                      cv::UMatUsageFlags usageFlags) const
    {
        return Allocator::allocate(data, accessflags, usageFlags);
    }

    template <class Allocator>
    void LockPolicyImpl<Allocator, cv::Mat>::deallocate(cv::UMatData* data) const
    {
        return Allocator::deallocate(data);
    }

    template <class Allocator>
    unsigned char* LockPolicyImpl<Allocator, cv::Mat>::allocate(size_t num_bytes)
    {
        boost::mutex::scoped_lock lock(m_mtx);
        return Allocator::allocate(num_bytes);
    }

    template <class Allocator>
    void LockPolicyImpl<Allocator, cv::Mat>::deallocate(unsigned char* ptr, size_t num_bytes)
    {
        boost::mutex::scoped_lock lock(m_mtx);
        return Allocator::deallocate(ptr, num_bytes);
    }

    /// ==========================================================
    /// RefCountPolicy
    template <class Allocator>
    RefCountPolicyImpl<Allocator, cv::Mat>::~RefCountPolicyImpl()
    {
        CV_Assert(m_ref_count == 0);
    }

    template <class Allocator>
    cv::UMatData* RefCountPolicyImpl<Allocator, cv::Mat>::allocate(
        int dims, const int* sizes, int type, void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const
    {
        auto ret = Allocator::allocate(dims, sizes, type, data, step, flags, usageFlags);
        if (ret)
        {
            return ret;
        }
        return nullptr;
    }

    template <class Allocator>
    bool RefCountPolicyImpl<Allocator, cv::Mat>::allocate(cv::UMatData* data,
                                                          int accessflags,
                                                          cv::UMatUsageFlags usageFlags) const
    {
        if (Allocator::allocate(data, accessflags, usageFlags))
        {
            return true;
        }
        return false;
    }

    template <class Allocator>
    void RefCountPolicyImpl<Allocator, cv::Mat>::deallocate(cv::UMatData* data) const
    {
        Allocator::deallocate(data);
    }

    template <class Allocator>
    unsigned char* RefCountPolicyImpl<Allocator, cv::Mat>::allocate(size_t num_bytes)
    {
        ++m_ref_count;
        return Allocator::allocate(num_bytes);
    }

    template <class Allocator>
    void RefCountPolicyImpl<Allocator, cv::Mat>::deallocate(unsigned char* ptr, size_t num_bytes)
    {
        --m_ref_count;
        Allocator::deallocate(ptr, num_bytes);
    }

    /// =========================================================
    /// GpuMat implementation
    template <class Allocator>
    RefCountPolicyImpl<Allocator, cv::cuda::GpuMat>::~RefCountPolicyImpl()
    {
        // CV_Assert(ref_count == 0 && "Warning, trying to delete allocator while cv::cuda::GpuMat's still reference
        // it");
        if (m_ref_count != 0)
        {
            MO_LOG(warning) << "Trying to delete allocator while " << m_ref_count
                            << " cv::cuda::GpuMats still reference it";
        }
    }

    template <class Allocator>
    bool RefCountPolicyImpl<Allocator, cv::cuda::GpuMat>::allocate(cv::cuda::GpuMat* mat,
                                                                   int rows,
                                                                   int cols,
                                                                   size_t elemSize)
    {
        if (Allocator::allocate(mat, rows, cols, elemSize))
        {
            ++m_ref_count;
            return true;
        }
        return false;
    }

    template <class Allocator>
    void RefCountPolicyImpl<Allocator, cv::cuda::GpuMat>::free(cv::cuda::GpuMat* mat)
    {
        Allocator::free(mat);
        --m_ref_count;
    }

    template <class Allocator>
    unsigned char* RefCountPolicyImpl<Allocator, cv::cuda::GpuMat>::allocate(size_t num_bytes)
    {
        if (auto ptr = Allocator::allocate(num_bytes))
        {
            ++m_ref_count;
            return ptr;
        }
        return nullptr;
    }

    template <class Allocator>
    void RefCountPolicyImpl<Allocator, cv::cuda::GpuMat>::deallocate(unsigned char* ptr, size_t num_bytes)
    {
        Allocator::deallocate(ptr, num_bytes);
        --m_ref_count;
    }

    /// ==========================================================
    /// ScopedDebugPolicy

    template <class Allocator>
    bool
    ScopeDebugPolicy<Allocator, cv::cuda::GpuMat>::allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
    {
        if (Allocator::allocate(mat, rows, cols, elemSize))
        {
            scopeOwnership[mat->data] = getScopeName();
            scopedAllocationSize[getScopeName()] += mat->step * mat->rows;
            return true;
        }
        return false;
    }

    template <class Allocator>
    void ScopeDebugPolicy<Allocator, cv::cuda::GpuMat>::free(cv::cuda::GpuMat* mat)
    {
        Allocator::free(mat);
        auto itr = scopeOwnership.find(mat->data);
        if (itr != scopeOwnership.end())
        {
            scopedAllocationSize[itr->second] -= mat->rows * mat->step;
        }
    }

    template <class Allocator>
    unsigned char* ScopeDebugPolicy<Allocator, cv::cuda::GpuMat>::allocate(size_t num_bytes)
    {
        if (auto ptr = Allocator::allocate(num_bytes))
        {
            scopeOwnership[ptr] = getScopeName();
            scopedAllocationSize[getScopeName()] += num_bytes;
            return ptr;
        }
        return nullptr;
    }

    template <class Allocator>
    void ScopeDebugPolicy<Allocator, cv::cuda::GpuMat>::deallocate(unsigned char* ptr, size_t /*num_bytes*/)
    {
        Allocator::free(ptr);
        auto itr = scopeOwnership.find(ptr);
        if (itr != scopeOwnership.end())
        {
            scopedAllocationSize[itr->second] -= this->current_allocations[ptr];
        }
    }

    template <class SmallAllocator, class LargeAllocator>
    CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::cuda::GpuMat>::CombinedPolicyImpl(size_t threshold_)
        : threshold(threshold_)
    {
    }

    template <class SmallAllocator, class LargeAllocator>
    bool CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::cuda::GpuMat>::allocate(cv::cuda::GpuMat* mat,
                                                                                        int rows,
                                                                                        int cols,
                                                                                        size_t elemSize)
    {
        if (rows * cols * elemSize < threshold)
        {
            return SmallAllocator::allocate(mat, rows, cols, elemSize);
        }
        else
        {
            return LargeAllocator::allocate(mat, rows, cols, elemSize);
        }
    }

    template <class SmallAllocator, class LargeAllocator>
    void CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::cuda::GpuMat>::free(cv::cuda::GpuMat* mat)
    {
        if (mat->rows * mat->cols * mat->elemSize() < threshold)
        {
            SmallAllocator::free(mat);
        }
        else
        {
            LargeAllocator::free(mat);
        }
    }

    template <class SmallAllocator, class LargeAllocator>
    unsigned char* CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::cuda::GpuMat>::allocate(size_t num_bytes)
    {
        return SmallAllocator::allocate(num_bytes);
    }

    template <class SmallAllocator, class LargeAllocator>
    void CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::cuda::GpuMat>::deallocate(unsigned char* ptr,
                                                                                          size_t num_bytes)
    {
        return SmallAllocator::deallocate(ptr, num_bytes);
    }

    template <class SmallAllocator, class LargeAllocator>
    void CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::cuda::GpuMat>::release()
    {
        SmallAllocator::release();
        LargeAllocator::release();
    }

    template <class SmallAllocator, class LargeAllocator>
    CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::Mat>::CombinedPolicyImpl(size_t threshold_)
        : threshold(threshold_)
    {
    }

    template <class SmallAllocator, class LargeAllocator>
    cv::UMatData* CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::Mat>::allocate(
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
        if (total < threshold)
        {
            return SmallAllocator::allocate(dims, sizes, type, data, step, flags, usageFlags);
        }
        else
        {
            return LargeAllocator::allocate(dims, sizes, type, data, step, flags, usageFlags);
        }
    }

    template <class SmallAllocator, class LargeAllocator>
    bool CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::Mat>::allocate(cv::UMatData* data,
                                                                               int /*accessflags*/,
                                                                               cv::UMatUsageFlags /*usageFlags*/) const
    {
        return (data != nullptr);
    }

    template <class SmallAllocator, class LargeAllocator>
    void CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::Mat>::deallocate(cv::UMatData* data) const
    {
        if (data->size < threshold)
        {
            SmallAllocator::deallocate(data);
        }
        else
        {
            LargeAllocator::deallocate(data);
        }
    }

    template <class SmallAllocator, class LargeAllocator>
    unsigned char* CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::Mat>::allocate(size_t num_bytes)
    {
        if (num_bytes < threshold)
        {
            return SmallAllocator::allocate(num_bytes);
        }
        else
        {
            return LargeAllocator::allocate(num_bytes);
        }
    }

    template <class SmallAllocator, class LargeAllocator>
    void CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::Mat>::deallocate(unsigned char* ptr, size_t num_bytes)
    {
        if (num_bytes < threshold)
        {
            return SmallAllocator::deallocate(ptr, num_bytes);
        }
        else
        {
            return LargeAllocator::deallocate(ptr, num_bytes);
        }
    }

    template <class SmallAllocator, class LargeAllocator>
    void CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::Mat>::release()
    {
        SmallAllocator::release();
        LargeAllocator::release();
    }

    template <class SmallAllocator, class LargeAllocator>
    CombinedPolicy<SmallAllocator, LargeAllocator>::CombinedPolicy(size_t threshold)
        : CombinedPolicyImpl<SmallAllocator, LargeAllocator, typename LargeAllocator::MatType>(threshold)
    {
    }

    template <class CPUAllocator, class GPUAllocator>
    class ConcreteAllocator : virtual public GPUAllocator, virtual public CPUAllocator, virtual public mo::Allocator
    {
      public:
        // GpuMat allocate
        bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
        {
            return GPUAllocator::allocate(mat, rows, cols, elemSize);
        }

        void free(cv::cuda::GpuMat* mat) { return GPUAllocator::free(mat); }

        // Thrust allocate
        unsigned char* allocateGpu(size_t num_bytes) { return GPUAllocator::allocate(num_bytes); }

        void deallocateGpu(unsigned char* ptr, size_t num_bytes) { return GPUAllocator::deallocate(ptr, num_bytes); }

        unsigned char* allocateCpu(size_t num_bytes) { return CPUAllocator::allocate(num_bytes); }

        void deallocateCpu(unsigned char* ptr, size_t num_bytes) { CPUAllocator::deallocate(ptr, num_bytes); }

        // CPU allocate
        cv::UMatData* allocate(int dims,
                               const int* sizes,
                               int type,
                               void* data,
                               size_t* step,
                               int flags,
                               cv::UMatUsageFlags usageFlags) const
        {
            return CPUAllocator::allocate(dims, sizes, type, data, step, flags, usageFlags);
        }

        bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const
        {
            return CPUAllocator::allocate(data, accessflags, usageFlags);
        }

        void deallocate(cv::UMatData* data) const { CPUAllocator::deallocate(data); }

        void release()
        {
            CPUAllocator::release();
            GPUAllocator::release();
        }
    };

    template <class CPUAllocator>
    class ConcreteCPUAllocator : virtual public CPUAllocator, virtual public mo::Allocator
    {
      public:
        // GpuMat allocate
        bool allocate(cv::cuda::GpuMat*, int , int , size_t )
        {
            return false;
        }

        void free(cv::cuda::GpuMat* ) { }

        // Thrust allocate
        unsigned char* allocateGpu(size_t ) { return nullptr; }

        void deallocateGpu(unsigned char* , size_t ) { }

        unsigned char* allocateCpu(size_t num_bytes) { return CPUAllocator::allocate(num_bytes); }

        void deallocateCpu(unsigned char* ptr, size_t num_bytes) { CPUAllocator::deallocate(ptr, num_bytes); }

        // CPU allocate
        cv::UMatData* allocate(int dims,
                               const int* sizes,
                               int type,
                               void* data,
                               size_t* step,
                               int flags,
                               cv::UMatUsageFlags usageFlags) const
        {
            return CPUAllocator::allocate(dims, sizes, type, data, step, flags, usageFlags);
        }

        bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const
        {
            return CPUAllocator::allocate(data, accessflags, usageFlags);
        }

        void deallocate(cv::UMatData* data) const { CPUAllocator::deallocate(data); }

        void release()
        {
            CPUAllocator::release();
        }
    };

    typedef PoolPolicy<cv::cuda::GpuMat, PitchedPolicy> d_TensorPoolAllocator_t;
    typedef LockPolicy<d_TensorPoolAllocator_t> d_mt_TensorPoolAllocator_t;

    typedef StackPolicy<cv::cuda::GpuMat, PitchedPolicy> d_TensorAllocator_t;
    typedef StackPolicy<cv::cuda::GpuMat, PitchedPolicy> d_TextureAllocator_t;

    typedef LockPolicy<d_TensorAllocator_t> d_mt_TensorAllocator_t;
    typedef LockPolicy<d_TextureAllocator_t> d_mt_TextureAllocator_t;

    typedef CpuPoolPolicy h_PoolAllocator_t;
    typedef CpuStackPolicy h_StackAllocator_t;

    typedef mt_CpuPoolPolicy h_mt_PoolAllocator_t;
    typedef mt_CpuStackPolicy h_mt_StackAllocator_t;

    typedef RefCountPolicy<CombinedPolicy<d_TensorPoolAllocator_t, d_TextureAllocator_t>> d_UniversalAllocator_t;
    typedef RefCountPolicy<LockPolicy<d_UniversalAllocator_t>> d_mt_UniversalAllocator_t;

    typedef CombinedPolicy<h_PoolAllocator_t, h_StackAllocator_t> h_UniversalAllocator_t;
    typedef LockPolicy<h_UniversalAllocator_t> h_mt_UniversalAllocator_t;

    typedef ConcreteAllocator<h_mt_PoolAllocator_t, d_mt_TensorPoolAllocator_t> mt_TensorAllocator_t;
    typedef ConcreteAllocator<h_PoolAllocator_t, d_TensorAllocator_t> TensorAllocator_t;

    typedef ConcreteAllocator<h_mt_StackAllocator_t, d_mt_TextureAllocator_t> mt_TextureAllocator_t;
    typedef ConcreteAllocator<h_StackAllocator_t, d_TextureAllocator_t> TextureAllocator_t;

    typedef ConcreteAllocator<h_UniversalAllocator_t, d_UniversalAllocator_t> UniversalAllocator_t;
    typedef ConcreteAllocator<h_mt_UniversalAllocator_t, d_mt_UniversalAllocator_t> mt_UniversalAllocator_t;

    typedef ConcreteCPUAllocator<h_mt_UniversalAllocator_t> mt_CPUAllocator_t;
}
