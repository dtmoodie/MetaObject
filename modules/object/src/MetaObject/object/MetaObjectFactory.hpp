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

    class MO_EXPORTS MetaObjectFactory
    {
      public:
        using Ptr_t = std::shared_ptr<MetaObjectFactory>;

        enum PluginVerbosity
        {
            brief, // plugin path
            info,  // brief + load info (load time and load state)
            debug  // info + build info
        };

        MO_INLINE static Ptr_t instance();
        static Ptr_t instance(SystemTable* table);

        MetaObjectFactory() = default;
        virtual ~MetaObjectFactory();

        virtual rcc::shared_ptr<IMetaObject> create(const char* type_name, int64_t interface_id = -1) = 0;
        template <class T>
        rcc::shared_ptr<T> create(const char* type_name);

        virtual rcc::shared_ptr<IMetaObject> get(ObjectId id, const char* type_name) = 0;

        virtual std::vector<std::string> listConstructableObjects(int interface_id = -1) const = 0;
        virtual std::string printAllObjectInfo(int64_t interface_id = -1) const = 0;

        virtual std::vector<IObjectConstructor*> getConstructors(int64_t interface_id = -1) const = 0;
        virtual IObjectConstructor* getConstructor(const char* type_name) const = 0;
        virtual const IObjectInfo* getObjectInfo(const char* type_name) const = 0;
        virtual std::vector<const IObjectInfo*> getAllObjectInfo() const = 0;

        virtual bool loadPlugin(const std::string& filename) = 0;
        virtual int loadPlugins(const std::string& path = "./") = 0;

        virtual std::vector<std::string> listLoadedPlugins(PluginVerbosity verbosity = brief) const = 0;
        virtual std::vector<PluginInfo> listLoadedPluginInfo() const = 0;

        // This function is inlined to guarantee it exists in the calling translation unit, which
        // thus makes certain to load the correct PerModuleInterface instance
        MO_INLINE void registerTranslationUnit();
        virtual void setupObjectConstructors(IPerModuleInterface* pPerModuleInterface) = 0;
        virtual IRuntimeObjectSystem* getObjectSystem() = 0;

        // Recompilation stuffs
        virtual bool abortCompilation() = 0;
        virtual bool checkCompile() = 0;
        virtual bool isCurrentlyCompiling() = 0;
        virtual bool isCompileComplete() = 0;
        virtual bool swapObjects() = 0;
        virtual void setCompileCallback(std::function<void(const std::string, int)>& f) = 0;
        virtual std::shared_ptr<Connection> connectConstructorAdded(TSlot<void(void)>& slot) = 0;

        template <class T>
        std::vector<IObjectConstructor*> getConstructors();

        template <class T>
        std::vector<typename T::InterfaceInfo*> getObjectInfos();

        template <class TINFO, class SCORING_FUNCTOR>
        IObjectConstructor* selectBestObjectConstructor(int64_t interface_id, SCORING_FUNCTOR&& scorer) const;

        template <class T, class TINFO, class SCORING_FUNCTOR>
        rcc::shared_ptr<T> createBestObject(int64_t interface_id, SCORING_FUNCTOR&& scorer) const;
    };

    MO_INLINE std::shared_ptr<MetaObjectFactory> MetaObjectFactory::instance()
    {
        return singleton<MetaObjectFactory>();
    }

    template <class T>
    rcc::shared_ptr<T> MetaObjectFactory::create(const char* type_name)
    {
        return create(type_name, T::s_interfaceID);
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
            const auto* info = dynamic_cast<const typename T::InterfaceInfo*>(constructor->GetObjectInfo());
            if (info)
            {
                output.push_back(info);
            }
        }
        return output;
    }

    MO_INLINE void MetaObjectFactory::registerTranslationUnit()
    {
        auto module = PerModuleInterface::GetInstance();
        setupObjectConstructors(module);
    }

    template <class TINFO, class SCORING_FUNCTOR>
    IObjectConstructor* MetaObjectFactory::selectBestObjectConstructor(int64_t interface_id,
                                                                       SCORING_FUNCTOR&& scorer) const
    {
        IObjectConstructor* best = nullptr;
        int32_t priority = 0;
        const auto ctrs = getConstructors(interface_id);

        for (auto ctr : ctrs)
        {
            const auto info = ctr->GetObjectInfo();
            const auto tinfo = dynamic_cast<const TINFO*>(info);
            if (tinfo)
            {
                auto score = scorer(*tinfo);
                if (score > priority)
                {
                    best = ctr;
                    priority = score;
                }
            }
        }
        return best;
    }

    template <class T, class TINFO, class SCORING_FUNCTOR>
    rcc::shared_ptr<T> MetaObjectFactory::createBestObject(int64_t interface_id, SCORING_FUNCTOR&& scorer) const
    {
        auto ctr = selectBestObjectConstructor<TINFO>(interface_id, scorer);
        if (ctr)
        {
            auto obj = ctr->Construct();
            if (obj)
            {
                obj->Init(true);
                return rcc::shared_ptr<T>(obj);
            }
        }
        return {};
    }

} // namespace mo
