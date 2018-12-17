#pragma once
#include "MetaObject/detail/Export.hpp"
#include <algorithm>
#include <unordered_map>
#include <vector>

namespace mo
{
    MO_EXPORTS const uint8_t* alignMemory(const uint8_t* ptr, const size_t elemSize);
    MO_EXPORTS uint8_t* alignMemory(uint8_t* ptr, const size_t elemSize);
    MO_EXPORTS size_t alignmentOffset(const uint8_t* ptr, const size_t elemSize);

    struct MO_EXPORTS CPU
    {
        static uint8_t* allocate(const size_t size, const size_t elem_size = 1);
        static void deallocate(uint8_t* data, const size_t size = 0);
    };

    template <class XPU>
    class MO_EXPORTS MemoryBlock
    {
      public:
        MemoryBlock(const size_t size_);
        ~MemoryBlock();

        uint8_t* allocate(const size_t size_, const size_t elemSize_);
        bool deallocate(uint8_t* ptr, const size_t num_elements);
        const uint8_t* begin() const;
        const uint8_t* end() const;
        uint8_t* begin();
        uint8_t* end();
        size_t size() const;

      protected:
        uint8_t* m_begin;
        uint8_t* m_end;
        std::unordered_map<uint8_t*, uint8_t*> m_allocated_blocks;
    };

    template <class T, class XPU>
    class TMemoryBlock : public MemoryBlock<XPU>
    {
      public:
        TMemoryBlock(const size_t num_elements);

        T* allocate(const size_t num_elements);
        bool deallocate(T* ptr, const size_t size);
        const T* begin() const;
        const T* end() const;
        T* begin();
        T* end();
        size_t size() const;
    };

    using CPUMemoryBlock = MemoryBlock<CPU>;
    extern template class MemoryBlock<CPU>;

    ////////////////////////////////////////////////////////////////////////////////
    ///                            MemoryBlock implementation
    ////////////////////////////////////////////////////////////////////////////////

    template <class XPU>
    MemoryBlock<XPU>::MemoryBlock(const size_t size_)
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
    uint8_t* MemoryBlock<XPU>::allocate(const size_t size_, const size_t elem_size_)
    {
        if (size_ > size())
        {
            return nullptr;
        }
        std::vector<std::pair<size_t, uint8_t*>> candidates;
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
        if (static_cast<size_t>(m_end - prev_end) >= size_)
        {
            auto alignment = alignmentOffset(prev_end, elem_size_);
            if (static_cast<size_t>(m_end - prev_end + alignment) >= size_)
            {
                candidates.emplace_back(size_t(m_end - prev_end + alignment), prev_end + alignment);
            }
        }
        // Find the smallest chunk of memory that fits our requirement, helps reduce fragmentation.
        auto min = std::min_element(
            candidates.begin(),
            candidates.end(),
            [](const std::pair<size_t, unsigned char*>& first, const std::pair<size_t, unsigned char*>& second) {
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
    bool MemoryBlock<XPU>::deallocate(uint8_t* ptr, const size_t /*size*/)
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
    size_t MemoryBlock<XPU>::size() const
    {
        return m_end - m_begin;
    }

    //////////////////////////////////////////////////////////////////////////
    ///   TMemoryBlock
    //////////////////////////////////////////////////////////////////////////

    template <class T, class XPU>
    TMemoryBlock<T, XPU>::TMemoryBlock(const size_t num_elements)
        : MemoryBlock<XPU>(num_elements * sizeof(T))
    {
    }

    template <class T, class XPU>
    T* TMemoryBlock<T, XPU>::allocate(const size_t num_elements)
    {
        return static_cast<T*>(static_cast<void*>(MemoryBlock<XPU>::allocate(num_elements * sizeof(T), sizeof(T))));
    }

    template <class T, class XPU>
    bool TMemoryBlock<T, XPU>::deallocate(T* ptr, const size_t num_elements)
    {
        return MemoryBlock<XPU>::deallocate(static_cast<uint8_t*>(static_cast<void*>(ptr)), num_elements * sizeof(T));
    }

    template <class T, class XPU>
    const T* TMemoryBlock<T, XPU>::begin() const
    {
        return static_cast<const T*>(static_cast<const void*>(MemoryBlock<XPU>::begin()));
    }

    template <class T, class XPU>
    const T* TMemoryBlock<T, XPU>::end() const
    {
        return static_cast<const T*>(static_cast<const void*>(MemoryBlock<XPU>::end()));
    }

    template <class T, class XPU>
    T* TMemoryBlock<T, XPU>::begin()
    {
        return static_cast<T*>(static_cast<void*>(MemoryBlock<XPU>::begin()));
    }

    template <class T, class XPU>
    T* TMemoryBlock<T, XPU>::end()
    {
        return static_cast<T*>(static_cast<void*>(MemoryBlock<XPU>::end()));
    }

    template <class T, class XPU>
    size_t TMemoryBlock<T, XPU>::size() const
    {
        return MemoryBlock<XPU>::size() / sizeof(T);
    }
}
