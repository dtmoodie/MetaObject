#pragma once
#include "MetaObject/detail/Export.hpp"
#include <algorithm>
#include <unordered_map>
#include <vector>

namespace mo
{
    MO_EXPORTS const uint8_t* alignMemory(const uint8_t* ptr, const uint64_t elemSize);
    MO_EXPORTS uint8_t* alignMemory(uint8_t* ptr, const uint64_t elemSize);
    MO_EXPORTS uint64_t alignmentOffset(const uint8_t* ptr, const uint64_t elemSize);

    struct MO_EXPORTS CPU
    {
        static uint8_t* allocate(const uint64_t size);
        static void deallocate(unsigned char* data);
    };

    template <class XPU>
    class MO_EXPORTS MemoryBlock
    {
      public:
        MemoryBlock(const uint64_t size_);
        ~MemoryBlock();

        uint8_t* allocate(const uint64_t size_, const uint64_t elemSize_);
        bool deAllocate(uint8_t* ptr, const uint64_t size);
        const uint8_t* begin() const;
        const uint8_t* end() const;
        uint8_t* begin();
        uint8_t* end();
        uint64_t size() const;

      protected:
        uint8_t* m_begin;
        uint8_t* m_end;
        std::unordered_map<uint8_t*, uint8_t*> m_allocated_blocks;
    };

    using CPUMemoryBlock = MemoryBlock<CPU>;
    extern template class MemoryBlock<CPU>;

    ////////////////////////////////////////////////////////////////////////////////
    ///                            MemoryBlock implementation
    ////////////////////////////////////////////////////////////////////////////////

    template <class XPU>
    MemoryBlock<XPU>::MemoryBlock(const uint64_t size_)
    {
        m_begin = XPU::allocate(size_);
        m_end = m_begin + size_;
    }

    template <class XPU>
    MemoryBlock<XPU>::~MemoryBlock()
    {
        XPU::deallocate(m_begin);
    }

    template <class XPU>
    uint8_t* MemoryBlock<XPU>::allocate(const uint64_t size_, const uint64_t elem_size_)
    {
        if (size_ > size())
        {
            return nullptr;
        }
        std::vector<std::pair<uint64_t, uint8_t*>> candidates;
        uint8_t* prev_end = m_begin;
        if (m_allocated_blocks.size())
        {
            for (auto itr : m_allocated_blocks)
            {
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
        if (static_cast<uint64_t>(m_end - prev_end) >= size_)
        {
            auto alignment = alignmentOffset(prev_end, elem_size_);
            if (static_cast<uint64_t>(m_end - prev_end + alignment) >= size_)
            {
                candidates.emplace_back(uint64_t(m_end - prev_end + alignment), prev_end + alignment);
            }
        }
        // Find the smallest chunk of memory that fits our requirement, helps reduce fragmentation.
        auto min = std::min_element(
            candidates.begin(),
            candidates.end(),
            [](const std::pair<uint64_t, unsigned char*>& first, const std::pair<uint64_t, unsigned char*>& second) {
                return first.first < second.first;
            });

        if (min != candidates.end() && min->first > size_)
        {
            m_allocated_blocks[min->second] = static_cast<uint8_t*>(min->second + size_);
            return min->second;
        }
        return nullptr;
    }

    template <class XPU>
    bool MemoryBlock<XPU>::deAllocate(uint8_t* ptr, const uint64_t /*size*/)
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
    const uint8_t* MemoryBlock<XPU>::begin() const
    {
        return m_begin;
    }

    template <class XPU>
    const uint8_t* MemoryBlock<XPU>::end() const
    {
        return m_end;
    }

    template <class XPU>
    uint8_t* MemoryBlock<XPU>::begin()
    {
        return m_begin;
    }

    template <class XPU>
    uint8_t* MemoryBlock<XPU>::end()
    {
        return m_end;
    }

    template <class XPU>
    uint64_t MemoryBlock<XPU>::size() const
    {
        return m_end - m_begin;
    }
}
