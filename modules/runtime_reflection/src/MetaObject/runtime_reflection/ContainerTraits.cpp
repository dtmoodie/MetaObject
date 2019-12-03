#include "ContainerTraits.hpp"

namespace mo
{

bool ContainerDefault::triviallySerializable() const
{
    return false;
}

TypeInfo ContainerDefault::keyType() const
{
    return TypeInfo::Void();
}

bool ContainerDefault::podKeys() const
{
    return true;
}

bool ContainerDefault::isContinuous() const
{
    return true;
}

size_t ContainerDefault::getContainerSize(const void*) const
{
    return 0;
}

void ContainerDefault::setContainerSize(size_t, void* ) const
{

}

void* ContainerDefault::valuePointer(void* inst) const
{
    return nullptr;
}

const void* ContainerDefault::valuePointer(const void* inst) const
{
    return nullptr;
}

void* ContainerDefault::keyPointer(void*) const
{
    return nullptr;
}

const void* ContainerDefault::keyPointer(const void*) const
{
    return nullptr;
}

}
