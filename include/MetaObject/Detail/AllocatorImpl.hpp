#pragma once
#include "Export.hpp"
#include <opencv2/core/cuda.hpp>

namespace mo
{
inline unsigned char* alignMemory(unsigned char* ptr, int elemSize)
{
    int i;
    for (i = 0; i < elemSize; ++i)
    {
        if (reinterpret_cast<size_t>(ptr + i) % elemSize == 0)
        {
            break;
        }
    }
    return ptr + i;  // Forces memory to be aligned to an element's byte boundary
}
inline int alignmentOffset(unsigned char* ptr, int elemSize)
{
    int i;
    for (i = 0; i < elemSize; ++i)
    {
        if (reinterpret_cast<size_t>(ptr + i) % elemSize == 0)
        {
            break;
        }
    }
    return i;
}
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
    PitchedPolicy()
    {
        textureAlignment = cv::cuda::DeviceInfo(cv::cuda::getDevice()).textureAlignment();
    }

    /*!
     * \brief SizeNeeded Calculates the data size needed for a 2D mat
     * \param rows [IN] Number of rows for 2D mat
     * \param cols [IN] Number of columns for 2D mat
     * \param elemSize [IN] Element size bytes
     * \param sizeNeeded [OUT] num bytes that needs to be allocated
     * \param stride [OUT] stride of a single row in bytes
     */
    inline void SizeNeeded(int rows, int cols, int elemSize,
                           size_t& sizeNeeded, size_t& stride)
    {
        if (rows == 1 || cols == 1)
        {
            stride = cols*elemSize;
        }
        else
        {
            if((cols*elemSize % textureAlignment) == 0)
                stride = cols*elemSize;
            else
                stride = cols*elemSize + textureAlignment - (cols*elemSize % textureAlignment);
        }
        sizeNeeded = stride*rows;
    }

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
    /*!
     * \brief SizeNeeded Calculates the data size needed for a 2D mat
     * \param rows [IN] Number of rows for 2D mat
     * \param cols [IN] Number of columns for 2D mat
     * \param elemSize [IN] Element size bytes
     * \param sizeNeeded [OUT] num bytes that needs to be allocated
     * \param stride [OUT] stride of a single row in bytes
     */
    inline void SizeNeeded(int rows, int cols, int elemSize,
                           size_t& sizeNeeded, size_t& stride)
    {
        stride = cols*elemSize;
        sizeNeeded = stride * rows;
    }
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
    inline size_t GetMemoryUsage() const
    {
        return memoryUsage;
    }
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
class MO_EXPORTS PoolPolicy {};

template<typename PaddingPolicy>
class MO_EXPORTS PoolPolicy<cv::cuda::GpuMat, PaddingPolicy>
        : public virtual AllocationPolicy
{
public:
    PoolPolicy(size_t initialBlockSize)
    {
        blocks.push_back(std::shared_ptr<GpuMemoryBlock>(new GpuMemoryBlock(_initial_block_size);
    }

    inline bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
    {
        size_t sizeNeeded, stride;
        PaddingPolicy::SizeNeeded(rows, cols, elemSize, sizeNeeded, stride);
        unsigned char* ptr;
        for (auto itr : blocks)
        {
            ptr = itr->allocate(sizeNeeded, elemSize);
            if (ptr)
            {
                mat->data = ptr;
                mat->step = stride;
                mat->refcount = (int*)cv::fastMalloc(sizeof(int));
                memoryUsage += mat->step*rows;
                LOG(trace) << "[GPU] Reusing block of size (" << rows << "," << cols << ") "
                           << mat->step * rows / (1024 * 1024) << " MB from memory block. Total usage: "
                           << memoryUsage / (1024 * 1024) << " MB";
                return true;
            }
        }
        // If we get to this point, then no memory was found, need to allocate new memory
        blocks.push_back(std::shared_ptr<GpuMemoryBlock>(
                             new GpuMemoryBlock(
                                 std::max(_initial_block_size / 2, sizeNeeded))));
        LOG(trace) << "[GPU] Expanding memory pool by " <<
                      std::max(_initial_block_size / 2, sizeNeeded) / (1024 * 1024)
                   << " MB";
        if (unsigned char* ptr = (*blocks.rbegin())->allocate(sizeNeeded, elemSize))
        {
            mat->data = ptr;
            mat->step = stride;
            mat->refcount = (int*)cv::fastMalloc(sizeof(int));
            memoryUsage += mat->step*rows;
            LOG(trace) << "[GPU] Reusing block of size (" << rows << "," << cols << ") "
                       << mat->step * rows / (1024 * 1024)
                       << " MB from memory block. Total usage: "
                       << memoryUsage / (1024 * 1024) << " MB";
            return true;
        }
        return false;
    }

    inline void free(cv::cuda::GpuMat* mat)
    {
        for (auto itr : blocks)
        {
            if (itr->deAllocate(mat->data))
            {
                cv::fastFree(mat->refcount);
                memoryUsage -= mat->step*mat->rows;
                return;
            }
        }
        throw cv::Exception(0, "[GPU] Unable to find memory to deallocate", __FUNCTION__, __FILE__, __LINE__);
    }

    inline unsigned char* allocate(size_t num_bytes)
    {
        unsigned char* ptr;
        for (auto itr : blocks)
        {
            ptr = itr->allocate(sizeNeeded, 1);
            if (ptr)
            {
                memoryUsage += sizeNeeded;

                return ptr;
            }
        }
        // If we get to this point, then no memory was found, need to allocate new memory
        blocks.push_back(std::shared_ptr<GpuMemoryBlock>(new GpuMemoryBlock(std::max(_initial_block_size / 2, sizeNeeded))));
        LOG(trace) << "[GPU] Expanding memory pool by "
                   << std::max(_initial_block_size / 2, sizeNeeded) / (1024 * 1024)
                   << " MB";
        if (unsigned char* ptr = (*blocks.rbegin())->allocate(sizeNeeded, 1))
        {
            memoryUsage += sizeNeeded;

            return ptr;
        }
        return nullptr;
    }

    inline void free(unsigned char* ptr)
    {
        for (auto itr : blocks)
        {
            if (itr->deAllocate(ptr))
            {
                return;
            }
        }
        throw cv::Exception(0, "[GPU] Unable to find memory to deallocate", __FUNCTION__, __FILE__, __LINE__);
    }

private:
    size_t _initial_block_size;
    std::list<std::shared_ptr<GpuMemoryBlock>> blocks;
};

template<>
class MO_EXPORTS PoolPolicy<cv::Mat, ContinuousPolicy>
{
public:

};

/*!
 *  \brief The StackPolicy class checks for a free memory stack of the exact
 *         requested size, if it is available it allocates from the free stack.
 *         If it wasn't able to find memory of the exact requested size, it will
 *         allocate the exact size and return it.  Since memory is not coelesced
 *         between stacks, this is best for large fixed size data, such as
 *         repeatedly allocated and deallocated images.
 */
template<typename T, typename PaddingPolicy>
class MO_EXPORTS StackPolicy{};

template<typename PaddingPolicy>
class MO_EXPORTS StackPolicy<cv::cuda::GpuMat, PaddingPolicy>
        : public virtual AllocationPolicy
{
public:
    bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
    void free(cv::cuda::GpuMat* mat);
    unsigned char* allocate(size_t num_bytes);
    void free(unsigned char* ptr);
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

/*!
 *  \brief The NonCachingPolicy allocates and deallocates the same as
 *         OpenCV's default allocator.  It has the advantage of memory
 *         usage tracking.
 */

template<typename T, typename PaddingPolicy>
class MO_EXPORTS NonCachingPolicy {};

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

/*!
 * \brief The LockPolicy class locks calls to the given allocator
 */
template<class Allocator>
class MO_EXPORTS LockPolicy: public virtual Allocator
{
public:
    inline bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
    inline void free(cv::cuda::GpuMat* mat);

    inline unsigned char* allocate(size_t num_bytes);
    inline void free(unsigned char* ptr);
private:
    boost::mutex mtx;
};

/*!
 * \brief The ScopeDebugPolicy class
 */
}
