#pragma once
#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"
#include "SystemTable.hpp"
#include <RuntimeObjectSystem/shared_ptr.hpp>
#include <memory>
#include <type_traits>

namespace mo
{
    template <class T>
    typename std::enable_if<std::is_base_of<IObject, T>::value, rcc::shared_ptr<T>>::type getSingleton()
    {
        // TODO
        static rcc::weak_ptr<T> inst;
        rcc::shared_ptr<T> output = inst;
        if (!output)
        {
        }
    }

    template <class T, class... Args>
    typename std::enable_if<!std::is_base_of<IObject, T>::value, std::shared_ptr<T>>::type getSingleton(Args&&... args)
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

    template <class T>
    typename std::enable_if<!std::is_base_of<IObject, T>::value, std::shared_ptr<T>>::type getSystemTableSingleton()
    {
        auto table = PerModuleInterface::GetInstance()->GetSystemTable();
        auto instance = table->getSingleton<T>();
        if (!instance)
        {
            instance = std::make_shared<T>();
            table->setSingleton(instance);
        }
        return instance.get();
    }

    template <class T>
    typename std::enable_if<std::is_base_of<IObject, T>::value, rcc::shared_ptr<T>>::type getSystemTableSingleton()
    {
        // TODO
        /*auto table = PerModuleInterface::GetInstance()->GetSystemTable();
        auto instance = table->getSingleton<T>();
        if (!instance)
        {
            instance = std::make_shared<T>();
            table->setSingleton(instance);
        }
        return instance.get();*/
    }
}
