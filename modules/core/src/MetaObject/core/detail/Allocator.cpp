

#include "Allocator.hpp"
#include "MemoryBlock.hpp"
#include <MetaObject/core/SystemTable.hpp>

namespace mo
{
    std::shared_ptr<Allocator> Allocator::getDefault()
    {
        return SystemTable::instance()->getDefaultAllocator();
    }

    void Allocator::setDefault(std::shared_ptr<Allocator> alloc)
    {
        SystemTable::instance()->setDefaultAllocator(std::move(alloc));
    }

    uint8_t* alignMemory(uint8_t* ptr, const size_t elem_size)
    {
        return &ptr[alignmentOffset(ptr, elem_size)];
    }

    const uint8_t* alignMemory(const uint8_t* ptr, const size_t elem_size)
    {
        return &ptr[alignmentOffset(ptr, elem_size)];
    }

    size_t alignmentOffset(const uint8_t* ptr, const size_t elem_size)
    {
        return elem_size - (reinterpret_cast<const size_t>(ptr) % elem_size);
    }

    Allocator::~Allocator() = default;

    void Allocator::release()
    {
    }

    void Allocator::setName(const std::string& name)
    {
        m_name = name;
    }

    void Allocator::deallocate(void* ptr, size_t size)
    {
        deallocate(static_cast<uint8_t*>(ptr), size);
    }

    const std::string& Allocator::name() const
    {
        return m_name;
    }

    DeviceAllocator::~DeviceAllocator() = default;

    void DeviceAllocator::setName(const std::string& name)
    {
        m_name = name;
    }

    const std::string& DeviceAllocator::name() const
    {
        return m_name;
    }

    std::shared_ptr<DeviceAllocator> DeviceAllocator::getDefault()
    {
        return SystemTable::instance()->getSingletonOptional<DeviceAllocator>();
    }

    void DeviceAllocator::release()
    {
    }

    void DeviceAllocator::deallocate(void* ptr, size_t num_elems)
    {
        deallocate(ct::ptrCast<uint8_t>(ptr), num_elems);
    }

} // namespace mo
