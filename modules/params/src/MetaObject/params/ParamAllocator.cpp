/*#include "ParamAllocator.hpp"

namespace mo
{
    IParamDataAllocator::IParamDataAllocator(Allocator::Ptr_t alloc)
        : m_allocator(std::move(alloc))
    {
    }

    void IParamDataAllocator::setPaddingSize(std::size_t header, std::size_t footer)
    {
        m_header_size = header;
        m_footer_size = footer;
    }

    std::shared_ptr<mo::Allocator> IParamDataAllocator::getAllocator() const
    {
        return m_allocator;
    }

    void IParamDataAllocator::setAllocator(std::shared_ptr<mo::Allocator> allocator)
    {
        m_allocator = std::move(allocator);
    }

    IParamDataAllocator::AllocationData IParamDataAllocator::allocate(size_t num_bytes, size_t element_size)
    {
        // TODO check alignment with element_size
        num_bytes += (m_header_size + m_footer_size);
        IParamDataAllocator::AllocationData output;

        auto allocation = m_allocator->allocate(num_bytes, element_size);
        output.allocated_begin = allocation;
        output.allocated_data = allocation + m_header_size;
        output.allocated_end = allocation + num_bytes;
        return output;
    }

    void IParamDataAllocator::deallocate(uint8_t* ptr, size_t num_bytes)
    {
        m_allocator->deallocate(ptr, num_bytes);
    }

    void IParamDataAllocator::release()
    {
        m_allocator->release();
    }
}
*/
