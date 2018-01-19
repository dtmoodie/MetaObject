#pragma once
#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"
#include "SystemTable.hpp"
#include <RuntimeObjectSystem/shared_ptr.hpp>
#include <memory>
#include <type_traits>

namespace mo
{
    template <class Base, class Derived>
    constexpr bool is_base_of_v = std::is_base_of<Base, Derived>::value;

    template <class T>
    std::enable_if_t<is_base_of_v<IObject, T>, rcc::shared_ptr<T>> getSingleton()
    {
        // TODO
        static rcc::weak_ptr<T> inst;
        rcc::shared_ptr<T> output = inst;
        if (!output)
        {
        }
    }

    template <class T, class... Args>
    std::enable_if_t<!is_base_of_v<IObject, T>, std::shared_ptr<T>> getSingleton(Args&&... args)
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
    std::enable_if_t<!is_base_of_v<IObject, T>, std::shared_ptr<T>> getSystemTableSingleton()
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
    std::enable_if_t<is_base_of_v<IObject, T>, rcc::shared_ptr<T>> getSystemTableSingleton()
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
