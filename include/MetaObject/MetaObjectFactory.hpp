#pragma once
#include "MetaObject/Detail/Export.hpp"

struct SystemTable;
struct IRuntimeObjectSystem;
namespace mo
{
    class IMetaObject;
    class MO_EXPORTS MetaObjectFactory
    {
    public:
        IMetaObject* Create(const char* type_name);
        IRuntimeObjectSystem* GetObjectSystem();
        static MetaObjectFactory* Instance(SystemTable* system_table = nullptr);
    private:
        MetaObjectFactory(SystemTable* system_table);
        ~MetaObjectFactory();
        struct impl;
        impl* _pimpl;
    };
}