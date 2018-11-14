#include "Allocator.hpp"
#include "MemoryBlock.hpp"

namespace mo
{

    uint8_t* alignMemory(uint8_t* ptr, const uint64_t elem_size)
    {
        return &ptr[alignmentOffset(ptr, size_t(elem_size))];
    }

    const uint8_t* alignMemory(const uint8_t* ptr, const uint64_t elem_size)
    {
        return &ptr[alignmentOffset(ptr, size_t(elem_size))];
    }

    uint64_t alignmentOffset(const uint8_t* ptr, const uint64_t elem_size)
    {
        return elem_size - (reinterpret_cast<const size_t>(ptr) % elem_size);
    }

    Allocator::~Allocator()
    {
    }

    void Allocator::release()
    {
    }

    void Allocator::setName(const std::string& name)
    {
        m_name = name;
    }

    const std::string& Allocator::name() const
    {
        return m_name;
    }
} // namespace mo
