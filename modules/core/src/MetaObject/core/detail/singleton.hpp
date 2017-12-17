#pragma once
#include <memory>

namespace mo
{
    template<class T>
    T* uniqueSingleton()
    {
        static std::unique_ptr<T> inst = std::make_unique<T>();
        return inst.get();
    }

    template<class T>
    std::shared_ptr<T> sharedSingleton()
    {
        static std::weak_ptr<T> inst;
        std::shared_ptr<T> output = inst.lock();
        if(!output)
        {
            output = std::make_shared<T>();
            inst = output;
        }
        return output;
    }

    template<class T, class U = T>
    std::shared_ptr<U> sharedThreadSpecificSingleton()
    {
        static std::weak_ptr<T> inst;
        std::shared_ptr<U> output = inst.lock();
        if(!output)
        {
            auto tmp = std::make_shared<T>();
            inst = tmp;
            output = tmp;
        }
        return output;
    }
}