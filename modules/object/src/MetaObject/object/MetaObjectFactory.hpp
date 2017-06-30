#pragma once
#include "MetaObject/detail/Export.hpp"
#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"
#include <functional>
#include <memory>
struct SystemTable;
struct IRuntimeObjectSystem;
struct IObjectInfo;
struct IObjectConstructor;
namespace mo
{
    class IMetaObject;
    template<class Sig> class TSlot;
    class Connection;
    class MO_EXPORTS MetaObjectFactory
    {
    public:
        IMetaObject*                       create(const char* type_name, int interface_id = -1);
        template<class T> T*               create(const char* type_name);
        IMetaObject*                       get(ObjectId id, const char* type_name);

        static MetaObjectFactory*          instance(SystemTable* system_table = nullptr);

        std::vector<std::string>           listConstructableObjects(int interface_id = -1) const;
        std::string                        printAllObjectInfo(int64_t interface_id = -1) const;

        std::vector<IObjectConstructor*>   getConstructors(int64_t interface_id = -1) const;
        IObjectConstructor*                getConstructor(const char* type_name) const;
        IObjectInfo*                       getObjectInfo(const char* type_name) const;
        std::vector<IObjectInfo*>          getAllObjectInfo() const;

        bool                               loadPlugin(const std::string& filename);
        int                                loadPlugins(const std::string& path = "./");
        std::vector<std::string>           listLoadedPlugins() const;

        // This function is inlined to guarantee it exists in the calling translation unit, which
        // thus makes certain to load the correct PerModuleInterface instance
        inline void                        registerTranslationUnit()
        {
            setupObjectConstructors(PerModuleInterface::GetInstance());
        }
        void                               setupObjectConstructors(IPerModuleInterface* pPerModuleInterface);
        IRuntimeObjectSystem*              getObjectSystem();

        // Recompilation stuffs
        bool abortCompilation();
        bool checkCompile();
        bool isCurrentlyCompiling();
        bool isCompileComplete();
        bool swapObjects();
        void setCompileCallback(std::function<void(const std::string, int)>& f);
        std::shared_ptr<Connection> connectConstructorAdded(TSlot<void(void)>* slot);
        template<class T>
        std::vector<IObjectConstructor*> getConstructors()
        {
            return getConstructors(T::s_interfaceID);
        }
        template<class T>
        std::vector<typename T::InterfaceInfo*> getObjectInfos()
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
    private:
        MetaObjectFactory(SystemTable* system_table);
        ~MetaObjectFactory();
        struct impl;
        impl* _pimpl;
    };
    template<class T>
    T* MetaObjectFactory::create(const char* type_name)
    {
        return static_cast<T*>(create(type_name, T::s_interfaceID));
    }
}
