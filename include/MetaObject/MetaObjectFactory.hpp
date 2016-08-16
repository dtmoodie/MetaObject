#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "ObjectInterfacePerModule.h"
struct SystemTable;
struct IRuntimeObjectSystem;
struct IObjectInfo;
namespace mo
{
    class IMetaObject;
    class MO_EXPORTS MetaObjectFactory
    {
    public:
        IMetaObject* Create(const char* type_name);
        IRuntimeObjectSystem* GetObjectSystem();
        static MetaObjectFactory* Instance(SystemTable* system_table = nullptr);

        std::vector<std::string> ListConstructableObjects() const;
        IObjectInfo* GetObjectInfo(const char* type_name) const;

        void SetupObjectConstructors(IPerModuleInterface* pPerModuleInterface);

        bool LoadPlugin(const std::string& filename);
        std::vector<std::string> ListLoadedPlugins() const;

        // This function is inlined to guarantee it exists in the calling translation unit, which 
        // thus makes certain to load the correct PerModuleInterface instance
		inline void RegisterTranslationUnit()
		{
			SetupObjectConstructors(PerModuleInterface::GetInstance());
		}
    private:
        MetaObjectFactory(SystemTable* system_table);
        ~MetaObjectFactory();
        struct impl;
        impl* _pimpl;
    };
}