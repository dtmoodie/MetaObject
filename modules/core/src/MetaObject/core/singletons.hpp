#pragma once

#include <memory>
#include <type_traits>
#include <RuntimeObjectSystem/shared_ptr.hpp>

namespace mo
{
    template<class T>
    std::enable_if_t<std::is_base_of_v<IObject, T>, rcc::shared_ptr<T>> getSingleton()
    {
        static rcc::weak_ptr<T> inst;
        rcc::shared_ptr<T> output = inst;
        if (!output)
        {
            
        }
    }

    template<class T, class... Args>
    std::enable_if_t<!std::is_base_of_v<IObject, T>, std::shared_ptr<T>> getSingleton(Args&&... args)
    {
        static std::weak_ptr<T> inst;
        std::shared_ptr<T> output = inst.lock();
        if (!output)
        {
            output = std::make_shared<T>(std::forward<Args>(args)...);
            inst = output;
        }
        return output;
    }
}