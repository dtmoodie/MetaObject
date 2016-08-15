#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "IRuntimeObjectSystem.h"
#include "ObjectInterfacePerModule.h"
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
		inline void RegisterTranslationUnit()
		{
			GetObjectSystem()->SetupObjectConstructors(PerModuleInterface::GetInstance());
		}
    private:
        MetaObjectFactory(SystemTable* system_table);
        ~MetaObjectFactory();
        struct impl;
        impl* _pimpl;
    };
}