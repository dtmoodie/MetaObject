#include "TDataContainer.hpp"

namespace mo
{
    ParamAllocator::SerializationBuffer::SerializationBuffer(ParamAllocator& alloc, uint8_t* begin, size_t sz)
        : m_alloc(alloc)
        , ct::TArrayView<uint8_t>(begin, sz)
    {
    }

    ParamAllocator::SerializationBuffer::SerializationBuffer(ParamAllocator& alloc, uint8_t* begin, uint8_t* end)
        : m_alloc(alloc)
        , ct::TArrayView<uint8_t>(begin, end)
    {
    }

    ParamAllocator::SerializationBuffer::~SerializationBuffer()
    {
        m_alloc.deallocate(this->data());
    }

    ParamAllocator::Ptr_t ParamAllocator::create(Allocator::Ptr_t allocator)
    {
        return std::make_shared<ParamAllocator>(allocator);
    }

    ParamAllocator::ParamAllocator(Allocator::Ptr_t allocator)
        : m_allocator(std::move(allocator))
    {
    }

    ParamAllocator::~ParamAllocator()
    {
        if (m_allocator)
        {
            for (auto itr = m_allocations.begin(); itr != m_allocations.end(); ++itr)
            {
                m_allocator->deallocate(ptrCast<>(itr->begin), itr->end - itr->begin);
            }
        }
    }

    void ParamAllocator::setPadding(size_t header, size_t footer)
    {
        m_header_pad = header;
        m_footer_pad = footer;
    }

    ParamAllocator::SerializationBuffer
    ParamAllocator::allocateSerializationImpl(size_t header_sz, size_t footer_sz, const void* ptr, size_t elem_size)
    {
        // TODO add additional header padding such that alignment is maintained for the data segment
        size_t num = 0;
        for (auto& itr : m_allocations)
        {
            if (itr.requested == ptrCast<uint8_t>(ptr))
            {
                num = itr.requested_size;
                if (header_sz <= static_cast<size_t>(itr.requested - itr.begin) &&
                    footer_sz <= static_cast<size_t>(itr.end - (itr.requested + itr.requested_size * elem_size)))
                {
                    itr.ref_count++;
                    return SerializationBuffer(
                        *this, ptrCast<uint8_t>(itr.requested - header_sz), ptrCast<uint8_t>(itr.end));
                }
            }
        }
        setPadding(header_sz, footer_sz);

        auto allocation = allocateImpl(num, elem_size);
        const size_t allocated = allocation.end - allocation.begin;
        auto result = ptrCast<uint8_t>(allocation.begin);
        return SerializationBuffer(*this, result, allocated);
    }

    void ParamAllocator::deallocateImpl(void* ptr)
    {
        if (!m_allocator)
        {
            MO_LOG(error, "Root allocator has been cleaned up, cannot release memory to it");
            return;
        }
        for (auto itr = m_allocations.begin(); itr != m_allocations.end(); ++itr)
        {
            if (ptrCast<>(ptr) >= itr->begin && ptrCast<>(ptr) < itr->end)
            {
                itr->ref_count--;
                if (itr->ref_count == 0)
                {
                    m_allocator->deallocate(ptrCast<>(itr->begin), itr->end - itr->begin);
                    m_allocations.erase(itr);
                }
                return;
            }
        }
    }

    Allocator::Ptr_t ParamAllocator::getAllocator() const
    {
        return m_allocator;
    }

    void ParamAllocator::setAllocator(Allocator::Ptr_t allocator)
    {
        m_allocator = std::move(allocator);
    }

    ParamAllocator::CurrentAllocations ParamAllocator::allocateImpl(size_t num, size_t elem_size)
    {
        const size_t bytes = num * elem_size + m_header_pad + m_footer_pad;
        auto allocated = m_allocator->allocate(bytes, elem_size);
        CurrentAllocations allocation;
        allocation.begin = allocated;
        allocation.end = allocated + bytes;
        allocation.requested = allocated + m_header_pad;
        allocation.requested_size = num;
        m_allocations.push_back(allocation);
        return allocation;
    }
} // namespace mo
