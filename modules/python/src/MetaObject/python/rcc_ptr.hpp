#pragma once
#include <RuntimeObjectSystem/shared_ptr.hpp>

namespace rcc
{
    template <class T>
    T* get_pointer(const rcc::shared_ptr<T>& ptr)
    {
        return ptr.get();
    }

    template<class T>
    T* get_pointer(const rcc::weak_ptr<T>& ptr)
    {
        return ptr.get();
    }
}
