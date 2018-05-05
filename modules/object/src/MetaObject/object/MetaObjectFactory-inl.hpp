#pragma once
#include "MetaObject/core/SystemTable.hpp"

namespace mo
{

    class MetaObjectFactory;

    MO_INLINE MetaObjectFactory& MetaObjectFactory::instance()
    {
        MetaObjectFactory* ptr = nullptr;
        auto module = PerModuleInterface::GetInstance();
        if (module)
        {
            auto table = module->GetSystemTable();
            if (table)
            {
                ptr = table->metaobject_factory;
            }
        }
        MO_ASSERT(ptr);
        return *ptr;
    }

    template <class T>
    T* MetaObjectFactory::create(const char* type_name)
    {
        return static_cast<T*>(create(type_name, T::s_interfaceID));
    }

    template <class T>
    std::vector<IObjectConstructor*> MetaObjectFactory::getConstructors()
    {
        return getConstructors(T::s_interfaceID);
    }

    template <class T>
    std::vector<typename T::InterfaceInfo*> MetaObjectFactory::getObjectInfos()
    {
        auto constructors = getConstructors<T>();
        std::vector<typename T::InterfaceInfo*> output;
        for (auto constructor : constructors)
        {
            typename T::InterfaceInfo* info = dynamic_cast<typename T::InterfaceInfo*>(constructor->GetObjectInfo());
            if (info)
                output.push_back(info);
        }
        return output;
    }

    MO_INLINE void MetaObjectFactory::registerTranslationUnit()
    {
        setupObjectConstructors(PerModuleInterface::GetInstance());
    }
}
