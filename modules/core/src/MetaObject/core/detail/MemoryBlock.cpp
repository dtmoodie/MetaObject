#include "MetaObject/core/detail/MemoryBlock.hpp"
#include "MetaObject/core/detail/AllocatorImpl.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <opencv2/cudev/common.hpp>
#include <utility>
#include <vector>

using namespace mo;

void GPUMemory::_allocate(unsigned char** ptr, size_t size)
{
    CV_CUDEV_SAFE_CALL(cudaMalloc(ptr, size));
}
void GPUMemory::_deallocate(unsigned char* ptr)
{
    CV_CUDEV_SAFE_CALL(cudaFree(ptr));
}

void CPUMemory::_allocate(unsigned char** ptr, size_t size)
{
    CV_CUDEV_SAFE_CALL(cudaMallocHost(ptr, size));
}

void CPUMemory::_deallocate(unsigned char* ptr)
{
    CV_CUDEV_SAFE_CALL(cudaFreeHost(ptr));
}

template <class XPU>
MemoryBlock<XPU>::MemoryBlock(size_t size_)
{
    XPU::_allocate(&m_begin, size_);
    m_end = m_begin + size_;
}

template <class XPU>
MemoryBlock<XPU>::~MemoryBlock()
{
    XPU::_deallocate(m_begin);
}

template <class XPU>
unsigned char* MemoryBlock<XPU>::allocate(size_t size_, size_t elem_size_)
{
    if (size_ > size())
    {
        return nullptr;
    }
    std::vector<std::pair<size_t, unsigned char*>> candidates;
    unsigned char* prev_end = m_begin;
    if (m_allocated_blocks.size())
    {
        for (auto itr : m_allocated_blocks) {
            if (static_cast<size_t>(itr.first - prev_end) > size_)
            {
                auto alignment = alignmentOffset(prev_end, elem_size_);
                if (static_cast<size_t>(itr.first - prev_end + alignment) >= size_)
                {
                    candidates.emplace_back(size_t(itr.first - prev_end + alignment), prev_end + alignment);
                }
            }
            prev_end = itr.second;
        }
    }
    if (static_cast<size_t>(m_end - prev_end) >= size_)
    {
        auto alignment = alignmentOffset(prev_end, elem_size_);
        if (static_cast<size_t>(m_end - prev_end + alignment) >= size_)
        {
            candidates.emplace_back(size_t(m_end - prev_end + alignment), prev_end + alignment);
        }
    }
    // Find the smallest chunk of memory that fits our requirement, helps reduce fragmentation.
    auto min =
        std::min_element(candidates.begin(),
                         candidates.end(),
                         [](const std::pair<size_t, unsigned char*>& first,
                            const std::pair<size_t, unsigned char*>& second) { return first.first < second.first; });

    if (min != candidates.end() && min->first > size_)
    {
        m_allocated_blocks[min->second] = (unsigned char*)(min->second + size_);
        return min->second;
    }
    return nullptr;
}

template <class XPU>
bool MemoryBlock<XPU>::deAllocate(unsigned char* ptr)
{
    if (ptr < m_begin || ptr > m_end)
        return false;
    auto itr = m_allocated_blocks.find(ptr);
    if (itr != m_allocated_blocks.end())
    {
        m_allocated_blocks.erase(itr);
        return true;
    }
    return true;
}

template <class XPU>
const unsigned char* MemoryBlock<XPU>::begin() const
{
    return m_begin;
}

template <class XPU>
const unsigned char* MemoryBlock<XPU>::end() const
{
    return m_end;
}

template <class XPU>
unsigned char* MemoryBlock<XPU>::begin()
{
    return m_begin;
}

template <class XPU>
unsigned char* MemoryBlock<XPU>::end()
{
    return m_end;
}

template <class XPU>
size_t MemoryBlock<XPU>::size() const
{
    return m_end - m_begin;
}

template class MemoryBlock<CPUMemory>;
template class MemoryBlock<GPUMemory>;
