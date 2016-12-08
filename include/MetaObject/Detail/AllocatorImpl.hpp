#pragma once
#include "Allocator.hpp"
#include <boost/thread/tss.hpp>
#include <opencv2/cudev/common.hpp>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <boost/thread/mutex.hpp>
#include <boost/thread/tss.hpp>

namespace mo
{

unsigned char* alignMemory(unsigned char* ptr, int elemSize)
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
int alignmentOffset(unsigned char* ptr, int elemSize)
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

/// ==========================================================
/// PitchedPolicy
PitchedPolicy::PitchedPolicy()
{
    textureAlignment = cv::cuda::DeviceInfo(cv::cuda::getDevice()).textureAlignment();
}

void PitchedPolicy::SizeNeeded(int rows, int cols, int elemSize, size_t& sizeNeeded, size_t& stride)
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

/// ==========================================================
/// ContinuousPolicy
void ContinuousPolicy::SizeNeeded(int rows, int cols, int elemSize, size_t& sizeNeeded, size_t& stride)
{
    stride = cols*elemSize;
    sizeNeeded = stride * rows;
}

/// ==========================================================
/// PoolPolicy
template<typename PaddingPolicy>
bool PoolPolicy<cv::cuda::GpuMat, PaddingPolicy>::allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
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

template<typename PaddingPolicy>
void PoolPolicy<cv::cuda::GpuMat, PaddingPolicy>::free(cv::cuda::GpuMat* mat)
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

template<typename PaddingPolicy>
unsigned char* PoolPolicy<cv::cuda::GpuMat, PaddingPolicy>::allocate(size_t sizeNeeded)
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

template<typename PaddingPolicy>
void PoolPolicy<cv::cuda::GpuMat, PaddingPolicy>::free(unsigned char* ptr)
{
    for (auto itr : blocks)
    {
        if (itr->deAllocate(ptr))
        {
            return;
        }
    }
    throw cv::Exception(0, "[GPU] Unable to find memory to deallocate",
                        __FUNCTION__, __FILE__, __LINE__);
}

/// ==========================================================
/// StackPolicy
template<typename PaddingPolicy>
bool StackPolicy<cv::cuda::GpuMat, PaddingPolicy>::allocate(
        cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
{
    size_t sizeNeeded, stride;
    PaddingPolicy::SizeNeeded(rows, cols, elemSize, sizeNeeded, stride);
    for (auto itr = deallocateList.begin(); itr != deallocateList.end(); ++itr)
    {
        if(std::get<2>(*itr) == sizeNeeded)
        {
            mat->data = std::get<0>(*itr);
            mat->step = stride;
            mat->refcount = (int*)cv::fastMalloc(sizeof(int));
            deallocateList.erase(itr);
            LOG(trace) << "[GPU] Reusing block of size (" << rows << "," << cols << ") "
                       << mat->step * rows / (1024 * 1024) << " MB. total usage: " << memoryUsage / (1024 * 1024) << " MB";
            return true;
        }
    }
    if (rows > 1 && cols > 1)
    {
        CV_CUDEV_SAFE_CALL(cudaMallocPitch(&mat->data, &mat->step, elemSize * cols, rows));
        LOG(trace) << "[GPU] Allocating block of size (" << rows << "," << cols << ") "
                   << mat->step * rows / (1024 * 1024) << " MB. Total usage: " << memoryUsage / (1024 * 1024) << " MB";
    }
    else
    {
        CV_CUDEV_SAFE_CALL(cudaMalloc(&mat->data, elemSize * cols * rows));
        LOG(trace) << "[GPU] Allocating block of size (" << rows << "," << cols << ") "
                   << cols * rows / (1024 * 1024) << " MB. Total usage: " << memoryUsage / (1024 * 1024) << " MB";
        mat->step = elemSize * cols;
    }
    mat->refcount = (int*)cv::fastMalloc(sizeof(int));
    return true;
}

template<typename PaddingPolicy>
void StackPolicy<cv::cuda::GpuMat, PaddingPolicy>::free(cv::cuda::GpuMat* mat)
{

    this->memoryUsage -= mat->rows*mat->step;
    LOG(trace) << "[GPU] Releasing mat of size (" << mat->rows << ","
               << mat->cols << ") " << (mat->dataend - mat->datastart)/(1024*1024) << " MB to the memory pool";
    deallocateList.emplace_back(mat->data, clock(), mat->dataend - mat->datastart);
    cv::fastFree(mat->refcount);
    clear();
}

template<typename PaddingPolicy>
unsigned char* StackPolicy<cv::cuda::GpuMat, PaddingPolicy>::allocate(size_t sizeNeeded)
{
    unsigned char* ptr = nullptr;
    for (auto itr = deallocateList.begin(); itr != deallocateList.end(); ++itr)
    {
        if (std::get<2>(*itr) == sizeNeeded)
        {
            ptr = std::get<0>(*itr);
            deallocateList.erase(itr);
            memoryUsage += sizeNeeded;
            current_allocations[ptr] = sizeNeeded;
            return ptr;
        }
    }
    CV_CUDEV_SAFE_CALL(cudaMalloc(&ptr, sizeNeeded));
    this->memoryUsage += sizeNeeded;
    current_allocations[ptr] = sizeNeeded;
    return ptr;
}

template<typename PaddingPolicy>
void StackPolicy<cv::cuda::GpuMat, PaddingPolicy>::free(unsigned char* ptr)
{
    auto itr = current_allocations.find(ptr);
    if(itr != current_allocations.end())
    {
        current_allocations.erase(itr);
        deallocateList.emplace_back(ptr, clock(), current_allocations[ptr]);
    }

    clear();
}

template<typename PaddingPolicy>
void StackPolicy<cv::cuda::GpuMat, PaddingPolicy>::clear()
{
    auto time = clock();
    for (auto itr = deallocateList.begin(); itr != deallocateList.end(); ++itr)
    {
        if((time - std::get<1>(*itr)) > deallocateDelay)
        {
            memoryUsage -= std::get<2>(*itr);
            LOG(trace) << "[GPU] Deallocating block of size " << std::get<2>(*itr) /(1024*1024)
                       << "MB. Which was stale for " << time - std::get<1>(*itr) << " ms";
            CV_CUDEV_SAFE_CALL(cudaFree(std::get<0>(*itr)));
            itr = deallocateList.erase(itr);
        }
    }
}

/// ==========================================================
/// NonCachingPolicy
template<typename PaddingPolicy>
bool NonCachingPolicy<cv::cuda::GpuMat, PaddingPolicy>::allocate(
        cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
{
    size_t size_needed, stride;
    PaddingPolicy::SizeNeeded(rows, cols, elemSize, size_needed, stride);
    if (rows > 1 && cols > 1)
    {
        CV_CUDEV_SAFE_CALL(cudaMallocPitch(&mat->data, &mat->step, elemSize * cols, rows));
        memoryUsage += mat->step*rows;
        LOG(trace) << "[GPU] Allocating block of size (" << rows << "," << cols << ") "
                   << mat->step * rows / (1024 * 1024) << " MB. Total usage: "
                   << memoryUsage / (1024 * 1024) << " MB";
    }
    else
    {
        CV_CUDEV_SAFE_CALL(cudaMalloc(&mat->data, elemSize * cols * rows));
        memoryUsage += elemSize*cols*rows;
        LOG(trace) << "[GPU] Allocating block of size (" << rows << "," << cols << ") "
                   << cols * rows / (1024 * 1024) << " MB. Total usage: "
                   << memoryUsage / (1024 * 1024) << " MB";
        mat->step = elemSize * cols;
    }
    mat->refcount = (int*)cv::fastMalloc(sizeof(int));
    return true;
}

template<typename PaddingPolicy>
void NonCachingPolicy<cv::cuda::GpuMat, PaddingPolicy>::free(cv::cuda::GpuMat* mat)
{
    CV_CUDEV_SAFE_CALL(cudaFree(mat->data));
    cv::fastFree(mat->refcount);
}

template<typename PaddingPolicy>
unsigned char* NonCachingPolicy<cv::cuda::GpuMat, PaddingPolicy>::allocate(size_t num_bytes)
{
    unsigned char* ptr = nullptr;
    CV_CUDEV_SAFE_CALL(cudaMalloc(&ptr, num_bytes));
    memoryUsage += num_bytes;
    return ptr;
}

template<typename PaddingPolicy>
void NonCachingPolicy<cv::cuda::GpuMat, PaddingPolicy>::free(unsigned char* ptr)
{
    CV_CUDEV_SAFE_CALL(cudaFree(ptr));
}

/// ==========================================================
/// LockPolicy
template<class Allocator>
bool LockPolicy<Allocator>::allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
{
    boost::mutex::scoped_lock lock(mtx);
    return Allocator::allocate(mat, rows, cols, elemSize);
}

template<class Allocator>
void LockPolicy<Allocator>::free(cv::cuda::GpuMat* mat)
{
    boost::mutex::scoped_lock lock(mtx);
    return Allocator::free(mat);
}

template<class Allocator>
unsigned char* LockPolicy<Allocator>::allocate(size_t num_bytes)
{
    boost::mutex::scoped_lock lock(mtx);
    return Allocator::allocate(num_bytes);
}

template<class Allocator>
void LockPolicy<Allocator>::free(unsigned char* ptr)
{
    boost::mutex::scoped_lock lock(mtx);
    return Allocator::free(ptr);
}

/// ==========================================================
/// ScopedDebugPolicy
boost::thread_specific_ptr<std::string> current_scope;
void SetScopeName(const std::string& name)
{
    if(current_scope.get() == nullptr)
    {
        current_scope.reset(new std::string());
    }
    *current_scope = name;
}

const std::string& GetScopeName()
{
    if(current_scope.get() == nullptr)
    {
        current_scope.reset(new std::string());
    }
    return *current_scope;
}

template<class Allocator>
bool ScopeDebugPolicy<Allocator, cv::cuda::GpuMat>::allocate(
        cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
{
    if(Allocator::allocate(mat, rows, cols, elemSize))
    {
        scopeOwnership[mat->data] = GetScopeName();
        scopedAllocationSize[GetScopeName()] += mat->step * mat->rows;
        return true;
    }
    return false;
}

template<class Allocator>
void ScopeDebugPolicy<Allocator, cv::cuda::GpuMat>::free(cv::cuda::GpuMat* mat)
{
    Allocator::free(mat);
    auto itr = scopeOwnership.find(mat->data);
    if (itr != scopeOwnership.end())
    {
        scopedAllocationSize[itr->second] -= mat->rows * mat->step;
    }
}

template<class Allocator>
unsigned char* ScopeDebugPolicy<Allocator, cv::cuda::GpuMat>::allocate(size_t num_bytes)
{
    if(auto ptr = Allocator::allocate(num_bytes))
    {
        scopeOwnership[ptr] = GetScopeName();
        scopedAllocationSize[GetScopeName()] += num_bytes;
        return ptr;
    }
    return nullptr;
}

template<class Allocator>
void ScopeDebugPolicy<Allocator, cv::cuda::GpuMat>::free(unsigned char* ptr)
{
    Allocator::free(ptr);
    auto itr = scopeOwnership.find(ptr);
    if (itr != scopeOwnership.end())
    {
        scopedAllocationSize[itr->second] -= this->current_allocations[ptr];
    }
}
template<class CPUAllocator, class GPUAllocator>
class ConcreteAllocator: virtual public T, mo::Allocator
{
public:
    // GpuMat allocate
    bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
    {
        return GPUAllocator::allocate(mat, rows, cols, elemSize);
    }

    void free(cv::cuda::GpuMat* mat)
    {
        return GPUAllocator::free(mat);
    }

    // Thrust allocate
    unsigned char* allocate(size_t num_bytes)
    {
        return GPUAllocator::allocate(num_bytes);
    }

    void free(unsigned char* ptr)
    {
        return GPUAllocator::free(ptr);
    }

    // CPU allocate
    cv::UMatData* allocate(int dims, const int* sizes, int type,
        void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const
    {
        return CPUAllocator::allocate(dims, sizes, type, data,
                                      step, flags, usageFlags);
    }
    bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const
    {
        return CPUAllocator::allocate(data, accessflags, usageFlags);
    }
    void deallocate(cv::UMatData* data) const
    {
        CPUAllocator::deallocate(data);
    }
};
}
