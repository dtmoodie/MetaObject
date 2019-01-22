#pragma once

#include "MetaObject/detail/Export.hpp"
#include "MetaObject/detail/defines.hpp"
#include "MetaObject/logging/logging.hpp"

#include "MetaObject/core/SystemTable.hpp"

#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"

#include <functional>
#include <memory>

struct IRuntimeObjectSystem;
struct IObjectInfo;
struct IObjectConstructor;
struct SystemTable;

namespace mo
{
    class IMetaObject;
    template <class Sig>
    class TSlot;
    class Connection;

    struct MO_EXPORTS PluginInfo
    {
        PluginInfo(const std::string& path = "",
                   const std::string& state = "success",
                   unsigned int time = 0,
                   const char* info = nullptr,
                   unsigned int id = 0)
            : m_path(path)
            , m_state(state)
            , m_load_time(time)
            , m_build_info(info)
            , m_id(id)
        {
        }

        std::string getPath() const;
        std::string getState() const;
        unsigned int getLoadTime() const;
        std::string getBuildInfo() const;
        unsigned int getId() const;
        std::string getPluginName() const;

        std::string m_path;
        std::string m_state;
        unsigned int m_load_time = 0; // total ms to load plugin
        const char* m_build_info = nullptr;
        unsigned int m_id = 0;
    };

    template <>
    struct ObjectConstructor<MetaObjectFactory>
    {
        using SharedPtr_t = std::shared_ptr<MetaObjectFactory>;
        using UniquePtr_t = std::unique_ptr<MetaObjectFactory>;

        ObjectConstructor(SystemTable* table);

        SharedPtr_t createShared() const;

        UniquePtr_t createUnique() const;

        MetaObjectFactory* create() const;

      private:
        SystemTable* table;
    };

    class MO_EXPORTS MetaObjectFactory
    {
      public:
        using Ptr_t = std::shared_ptr<MetaObjectFactory>;

        MO_INLINE static Ptr_t instance();
        static Ptr_t instance(SystemTable* table);

        IMetaObject* create(const char* type_name, int64_t interface_id = -1);
        template <class T>
        T* create(const char* type_name);
        IMetaObject* get(ObjectId id, const char* type_name);

        std::vector<std::string> listConstructableObjects(int interface_id = -1) const;
        std::string printAllObjectInfo(int64_t interface_id = -1) const;

        std::vector<IObjectConstructor*> getConstructors(int64_t interface_id = -1) const;
        IObjectConstructor* getConstructor(const char* type_name) const;
        IObjectInfo* getObjectInfo(const char* type_name) const;
        std::vector<IObjectInfo*> getAllObjectInfo() const;

        bool loadPlugin(const std::string& filename);
        int loadPlugins(const std::string& path = "./");

        enum PluginVerbosity
        {
            brief, // plugin path
            info,  // brief + load info (load time and load state)
            debug  // info + build info
        };

        std::vector<std::string> listLoadedPlugins(PluginVerbosity verbosity = brief) const;
        std::vector<PluginInfo> listLoadedPluginInfo() const;

        // This function is inlined to guarantee it exists in the calling translation unit, which
        // thus makes certain to load the correct PerModuleInterface instance
        MO_INLINE void registerTranslationUnit();
        void setupObjectConstructors(IPerModuleInterface* pPerModuleInterface);
        IRuntimeObjectSystem* getObjectSystem();

        // Recompilation stuffs
        bool abortCompilation();
        bool checkCompile();
        bool isCurrentlyCompiling();
        bool isCompileComplete();
        bool swapObjects();
        void setCompileCallback(std::function<void(const std::string, int)>& f);
        std::shared_ptr<Connection> connectConstructorAdded(TSlot<void(void)>* slot);

        template <class T>
        std::vector<IObjectConstructor*> getConstructors();

        template <class T>
        std::vector<typename T::InterfaceInfo*> getObjectInfos();

        ~MetaObjectFactory();

      protected:
        friend struct ObjectConstructor<MetaObjectFactory>;

        MetaObjectFactory(SystemTable* table);

      private:
        struct impl;
        std::unique_ptr<impl> _pimpl;
    };

    MO_INLINE std::shared_ptr<MetaObjectFactory> MetaObjectFactory::instance()
    {
        std::shared_ptr<MetaObjectFactory> ptr;
        auto module = PerModuleInterface::GetInstance();
        if (module)
        {
            auto table = module->GetSystemTable();
            return instance(table);

        }
        MO_ASSERT(ptr);
        return ptr;
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
        auto module = PerModuleInterface::GetInstance();
        setupObjectConstructors(module);
    }

} // namespace mo
