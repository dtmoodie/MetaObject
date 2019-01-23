#include "MetaObject/object/MetaObjectFactory.hpp"
#include "MetaObject/core/SystemTable.hpp"
#include "MetaObject/core/detail/Time.hpp"
#include "MetaObject/logging/CompileLogger.hpp"
#include "MetaObject/logging/logging.hpp"
#include "MetaObject/object/IMetaObject.hpp"
#include "MetaObject/signals/TSignal.hpp"
#include "MetaObject/signals/TSlot.hpp"
#include "RuntimeObjectSystem/IObjectFactorySystem.h"
#include "RuntimeObjectSystem/RuntimeObjectSystem.h"
#include <MetaObject/thread/boost_thread.hpp>

#include <boost/date_time.hpp>
#include <boost/filesystem.hpp>
using namespace mo;

std::string PluginInfo::getPath() const
{
    return m_path;
}

std::string PluginInfo::getState() const
{
    return m_state;
}

unsigned int PluginInfo::getLoadTime() const
{
    return m_load_time;
}

std::string PluginInfo::getBuildInfo() const
{
    if (m_build_info)
        return std::string(m_build_info);
    return "";
}

unsigned int PluginInfo::getId() const
{
    return m_id;
}

std::string PluginInfo::getPluginName() const
{
    boost::filesystem::path path(m_path);
    return path.replace_extension("").filename().string();
}

class MetaObjectFactoryImpl: public MetaObjectFactory
{
  public:
    using Ptr_t = std::shared_ptr<MetaObjectFactory>;

    MetaObjectFactoryImpl(SystemTable* table)
    {
        obj_system.Initialise(&logger, table);
    }

    ~MetaObjectFactoryImpl() override
    {

    }

    IMetaObject* create(const char* type_name, int64_t interface_id = -1) override;

    virtual IMetaObject* get(ObjectId id, const char* type_name)  override;

    virtual std::vector<std::string> listConstructableObjects(int interface_id = -1) const  override;
    virtual std::string printAllObjectInfo(int64_t interface_id = -1) const  override;

    virtual std::vector<IObjectConstructor*> getConstructors(int64_t interface_id = -1) const  override;
    virtual IObjectConstructor* getConstructor(const char* type_name) const override;
    virtual IObjectInfo* getObjectInfo(const char* type_name) const  override;
    virtual std::vector<IObjectInfo*> getAllObjectInfo() const  override;

    virtual bool loadPlugin(const std::string& filename) override;
    virtual int loadPlugins(const std::string& path = "./") override;

    virtual std::vector<std::string> listLoadedPlugins(PluginVerbosity verbosity = brief) const override;;
    virtual std::vector<PluginInfo> listLoadedPluginInfo() const override;;

    // This function is inlined to guarantee it exists in the calling translation unit, which
    // thus makes certain to load the correct PerModuleInterface instance
    MO_INLINE void registerTranslationUnit();
    virtual void setupObjectConstructors(IPerModuleInterface* pPerModuleInterface) override;
    virtual IRuntimeObjectSystem* getObjectSystem() override;

    // Recompilation stuffs
    virtual bool abortCompilation() override;
    virtual bool checkCompile() override;
    virtual bool isCurrentlyCompiling() override;
    virtual bool isCompileComplete() override;
    virtual bool swapObjects() override;
    virtual void setCompileCallback(std::function<void(const std::string, int)>& f) override;
    virtual std::shared_ptr<Connection> connectConstructorAdded(TSlot<void(void)>* slot) override;
private:
    RuntimeObjectSystem obj_system;
    CompileLogger logger;
    std::vector<PluginInfo> plugins;
    TSignal<void(void)> on_constructor_added;
};

namespace mo
{
    template <>
    struct ObjectConstructor<MetaObjectFactoryImpl>
    {
        using SharedPtr_t = std::shared_ptr<MetaObjectFactoryImpl>;
        using UniquePtr_t = std::unique_ptr<MetaObjectFactoryImpl>;

        ObjectConstructor(SystemTable* table);

        SharedPtr_t createShared() const;

        UniquePtr_t createUnique() const;

        MetaObjectFactory* create() const;

      private:
        SystemTable* table;
    };
}

std::shared_ptr<MetaObjectFactory> MetaObjectFactory::instance(SystemTable* table)
{
    MO_ASSERT(table != nullptr);
    std::shared_ptr<MetaObjectFactory> ptr;
    auto module = PerModuleInterface::GetInstance();
    auto table_ = module->GetSystemTable();
    if(table_)
    {
        MO_ASSERT(table == table_);
    }else
    {
        module->SetSystemTable(table);
    }
    if (table)
    {
        ptr = sharedSingleton<MetaObjectFactory, MetaObjectFactoryImpl>(table, mo::ObjectConstructor<MetaObjectFactoryImpl>(table));
    }
    return ptr;
}

MetaObjectFactory::~MetaObjectFactory()
{
}

IRuntimeObjectSystem* MetaObjectFactoryImpl::getObjectSystem()
{
    return &obj_system;
}

IMetaObject* MetaObjectFactoryImpl::create(const char* type_name, int64_t interface_id)
{
    auto constructor = obj_system.GetObjectFactorySystem()->GetConstructor(type_name);
    if (constructor)
    {
        if (interface_id != -1)
        {
            if (constructor->GetInterfaceId() != interface_id)
                return nullptr;
        }
        IObject* obj = constructor->Construct();
        if (IMetaObject* mobj = dynamic_cast<IMetaObject*>(obj))
        {
            mobj->Init(true);
            return mobj;
        }
        else
        {
            delete obj;
        }
    }
    return nullptr;
}

IMetaObject* MetaObjectFactoryImpl::get(ObjectId id, const char* type_name)
{
    AUDynArray<IObjectConstructor*> constructors;
    obj_system.GetObjectFactorySystem()->GetAll(constructors);
    if (id.m_ConstructorId < constructors.Size())
    {
        if (strcmp(constructors[id.m_ConstructorId]->GetName(), type_name) == 0)
        {
            IObject* obj = constructors[id.m_ConstructorId]->GetConstructedObject(id.m_PerTypeId);
            if (obj)
            {
                return dynamic_cast<IMetaObject*>(obj);
            }
            else
            {
                MO_LOG(debug, "No object exits for {} with instance id of {}", type_name, id.m_PerTypeId);
            }
        }
        else
        {
            MO_LOG(debug,
                   "Requested type '{}' does not match constructor type '{}' for given ID",
                   type_name,
                   constructors[id.m_ConstructorId]->GetName());
            std::string str_type_name(type_name);
            for (size_t i = 0; i < constructors.Size(); ++i)
            {
                if (std::string(constructors[i]->GetName()) == str_type_name)
                {
                    IObject* obj = constructors[i]->GetConstructedObject(id.m_PerTypeId);
                    if (obj)
                    {
                        return dynamic_cast<IMetaObject*>(obj);
                    }
                    else
                    {
                        return nullptr; // Object just doesn't exist yet.
                    }
                }
            }
            MO_LOG(warn, "Requested type {} not found", type_name);
        }
    }
    return nullptr;
}

std::vector<std::string> MetaObjectFactoryImpl::listConstructableObjects(int interface_id) const
{
    std::vector<std::string> output;
    AUDynArray<IObjectConstructor*> constructors;
    obj_system.GetObjectFactorySystem()->GetAll(constructors);
    for (size_t i = 0; i < constructors.Size(); ++i)
    {
        if (interface_id == -1)
            output.emplace_back(constructors[i]->GetName());
        else if (constructors[i]->GetInterfaceId() == static_cast<uint32_t>(interface_id))
            output.emplace_back(constructors[i]->GetName());
    }
    return output;
}

std::string MetaObjectFactoryImpl::printAllObjectInfo(int64_t interface_id) const
{
    std::stringstream ss;
    AUDynArray<IObjectConstructor*> constructors;
    obj_system.GetObjectFactorySystem()->GetAll(constructors);
    for (size_t i = 0; i < constructors.Size(); ++i)
    {
        if (auto info = constructors[i]->GetObjectInfo())
        {
            if (interface_id == -1)
            {
                ss << info->Print();
            }
            else
            {
                if (constructors[i]->GetInterfaceId() == interface_id)
                {
                    ss << info->Print();
                }
            }
        }
    }
    return ss.str();
}

IObjectConstructor* MetaObjectFactoryImpl::getConstructor(const char* type_name) const
{
    return obj_system.GetObjectFactorySystem()->GetConstructor(type_name);
}

std::vector<IObjectConstructor*> MetaObjectFactoryImpl::getConstructors(int64_t interface_id) const
{
    std::vector<IObjectConstructor*> output;
    AUDynArray<IObjectConstructor*> constructors;
    obj_system.GetObjectFactorySystem()->GetAll(constructors);
    for (size_t i = 0; i < constructors.Size(); ++i)
    {
        if (interface_id == -1)
            output.emplace_back(constructors[i]);
        else if (constructors[i]->GetInterfaceId() == interface_id)
            output.emplace_back(constructors[i]);
    }
    return output;
}

IObjectInfo* MetaObjectFactoryImpl::getObjectInfo(const char* type_name) const
{
    auto constructor = obj_system.GetObjectFactorySystem()->GetConstructor(type_name);
    if (constructor)
    {
        return constructor->GetObjectInfo();
    }
    return nullptr;
}

std::vector<IObjectInfo*> MetaObjectFactoryImpl::getAllObjectInfo() const
{
    std::vector<IObjectInfo*> output;
    AUDynArray<IObjectConstructor*> constructors;
    obj_system.GetObjectFactorySystem()->GetAll(constructors);
    for (size_t i = 0; i < constructors.Size(); ++i)
    {
        output.push_back(constructors[i]->GetObjectInfo());
    }
    return output;
}

void MetaObjectFactoryImpl::setupObjectConstructors(IPerModuleInterface* pPerModuleInterface)
{
    getObjectSystem()->SetupObjectConstructors(pPerModuleInterface);
}

std::vector<std::string> MetaObjectFactoryImpl::listLoadedPlugins(PluginVerbosity verbosity) const
{
    std::vector<std::string> output;
    for (size_t i = 0; i < plugins.size(); ++i)
    {
        std::stringstream ss;
        switch (verbosity)
        {
        case brief:
            ss << plugins[i].m_path;
            break;
        case info:
            ss << plugins[i].m_path << " " << plugins[i].m_state << " ("
               << plugins[i].m_load_time << " ms)";
            break;
        case debug:
            ss << plugins[i].m_path << " " << plugins[i].m_state << " ("
               << plugins[i].m_load_time << " ms)";
            if (plugins[i].m_build_info)
                ss << "\n" << plugins[i].m_build_info;
            break;
        }
        output.push_back(ss.str());
    }
    return output;
}

std::vector<PluginInfo> MetaObjectFactoryImpl::listLoadedPluginInfo() const
{
    return plugins;
}

int MetaObjectFactoryImpl::loadPlugins(const std::string& path_)
{
    boost::filesystem::path path(boost::filesystem::current_path().string() + "/" + path_);
    int count = 0;
    for (boost::filesystem::directory_iterator itr(path); itr != boost::filesystem::directory_iterator(); ++itr)
    {
#ifdef _MSC_VER
        if (itr->path().extension().string() == ".dll")
#else
        if (itr->path().extension().string() == ".so")
#endif
        {
            if (loadPlugin(itr->path().string()))
                ++count;
        }
    }
    return count;
}

#ifdef _MSC_VER
#include "Windows.h"
std::string GetLastErrorAsString()
{
    // Get the error message, if any.
    DWORD errorMessageID = ::GetLastError();
    if (errorMessageID == 0)
        return std::string(); // No error message has been recorded

    LPSTR messageBuffer = nullptr;
    size_t size =
        FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                       NULL,
                       errorMessageID,
                       MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                       (LPSTR)&messageBuffer,
                       0,
                       NULL);

    std::string message(messageBuffer, size);

    // Free the buffer.
    LocalFree(messageBuffer);

    return message;
}

bool MetaObjectFactoryImpl::loadPlugin(const std::string& fullPluginPath)
{
    static int projectCount = 0;
    MO_LOG(info) << "Loading plugin " << fullPluginPath;
    if (!boost::filesystem::is_regular_file(fullPluginPath))
    {
        MO_LOG(warning) << fullPluginPath << " does not exist";
        return false;
    }
    PluginInfo plugin_info;
    std::string plugin_name = boost::filesystem::path(fullPluginPath).stem().string();
    plugin_info.m_path = fullPluginPath;
    mo::Time start = mo::getCurrentTime();
    HMODULE handle = LoadLibrary(fullPluginPath.c_str());
    if (handle == nullptr)
    {
        auto err = GetLastError();
        MO_LOG(warning) << "Failed to load " << plugin_name << " due to: [" << err << "] " << GetLastErrorAsString();
        plugin_info.m_state = "failed";
        plugins.push_back(plugin_info);
        return false;
    }
    typedef const char* (*InfoFunctor)();

    InfoFunctor info = (InfoFunctor)GetProcAddress(handle, "getPluginBuildInfo");
    if (info)
    {
        MO_LOG(debug) << info();
        plugin_info.m_build_info = info();
    }

    typedef IPerModuleInterface* (*moduleFunctor)();
    moduleFunctor module = (moduleFunctor)GetProcAddress(handle, "GetPerModuleInterface");
    if (module)
    {
        auto moduleInterface = module();
        moduleInterface->SetModuleFileName(fullPluginPath.c_str());
        boost::filesystem::path path(fullPluginPath);
        std::string base = path.stem().replace_extension("").string();
#ifdef _DEBUG
        base = base.substr(0, base.size() - 1);
#endif
        boost::filesystem::path config_path = path.parent_path();
        config_path += "/" + base + "_config.txt";
        int id = obj_system.ParseConfigFile(config_path.string().c_str());
        if (id >= 0)
        {
            moduleInterface->SetProjectIdForAllConstructors(id);
        }
        plugin_info.m_id = id;
        setupObjectConstructors(moduleInterface);
        mo::Time end = mo::getCurrentTime();
        plugin_info.m_state = "success";
        plugin_info.m_load_time =
            static_cast<unsigned int>(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        plugins.push_back(plugin_info);
        return true;
    }
    return false;
}
#else
#include "dlfcn.h"

bool MetaObjectFactoryImpl::loadPlugin(const std::string& fullPluginPath)
{
    std::string old_name = mo::getThisThreadName();
    MO_LOG(info, "Loading {}", fullPluginPath);
    boost::filesystem::path path(fullPluginPath);
    if(!boost::filesystem::exists(path))
    {
        MO_LOG(warn, "{} does not exist", fullPluginPath);
        return false;
    }
    std::string base = path.stem().replace_extension("").string();
    mo::setThisThreadName(base.substr(3));
    boost::posix_time::ptime start = boost::posix_time::microsec_clock::universal_time();
    void* handle = dlopen(fullPluginPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    // Fallback on old module
    if (handle == nullptr)
    {
        const char* dlsym_error = dlerror();
        if (dlsym_error)
        {
            MO_LOG(warn, dlsym_error);
            boost::posix_time::ptime end = boost::posix_time::microsec_clock::universal_time();
            plugins.emplace_back(fullPluginPath,
                                         "failed (dlsym_error - " + std::string(dlsym_error),
                                         (end - start).total_milliseconds(),
                                         nullptr);
            mo::setThisThreadName(old_name);
            return false;
        }
    }

    typedef void (*InitFunctor)();

    InitFunctor init = InitFunctor(dlsym(handle, "InitModule"));
    if (init)
    {
        init();
    }
    typedef const char* (*InfoFunctor)();
    InfoFunctor info = InfoFunctor(dlsym(handle, "getPluginBuildInfo"));
    if (info)
    {
        MO_LOG(debug, info());
    }

    typedef IPerModuleInterface* (*moduleFunctor)();

    moduleFunctor module = moduleFunctor(dlsym(handle, "GetPerModuleInterface"));
    const char* dlsym_error = dlerror();
    if (dlsym_error)
    {
        MO_LOG(warn, dlsym_error);
        boost::posix_time::ptime end = boost::posix_time::microsec_clock::universal_time();
        plugins.emplace_back(fullPluginPath,
                                     "failed (dlsym_error - " + std::string(dlsym_error),
                                     (end - start).total_milliseconds(),
                                     info ? info() : nullptr);
        mo::setThisThreadName(old_name);
        return false;
    }
    if (module == nullptr)
    {
        MO_LOG(warn, "module == nullptr");
        boost::posix_time::ptime end = boost::posix_time::microsec_clock::universal_time();
        plugins.emplace_back(
            fullPluginPath, "failed (module == nullptr)", (end - start).total_milliseconds(), info ? info() : nullptr);
        mo::setThisThreadName(old_name);
        return false;
    }
    IPerModuleInterface* interface = module();
    interface->SetModuleFileName(fullPluginPath.c_str());

#ifdef NDEBUG
    base = base.substr(3, base.size() - 3);
#else
    base = base.substr(3, base.size() - 4); // strip off the d tag on the library file
#endif
    boost::filesystem::path config_path = path.parent_path();
    config_path += "/" + base + "_config.txt";
    int id = obj_system.ParseConfigFile(config_path.string().c_str());

    if (id >= 0)
    {
#ifdef _DEBUG
        obj_system.SetOptimizationLevel(RCCPPOPTIMIZATIONLEVEL_DEBUG, id);
#else
        _pimpl->obj_system.SetOptimizationLevel(RCCPPOPTIMIZATIONLEVEL_PERF, id);
#endif
        interface->SetProjectIdForAllConstructors(static_cast<unsigned short>(id));
    }
    setupObjectConstructors(interface);
    boost::posix_time::ptime end = boost::posix_time::microsec_clock::universal_time();
    plugins.emplace_back(
        fullPluginPath, "success", (end - start).total_milliseconds(), info ? info() : nullptr, id);
    mo::setThisThreadName(old_name);
    return true;
}

#endif

bool MetaObjectFactoryImpl::abortCompilation()
{
    return obj_system.AbortCompilation();
}

bool MetaObjectFactoryImpl::checkCompile()
{
    static boost::posix_time::ptime prevTime = boost::posix_time::microsec_clock::universal_time();
    boost::posix_time::ptime currentTime = boost::posix_time::microsec_clock::universal_time();
    boost::posix_time::time_duration delta = currentTime - prevTime;
    if (delta.total_milliseconds() < 10)
        return false;
    obj_system.GetFileChangeNotifier()->Update(float(delta.total_milliseconds()) / 1000.0f);
    return isCurrentlyCompiling();
}

bool MetaObjectFactoryImpl::isCurrentlyCompiling()
{
    return obj_system.GetIsCompiling();
}

bool MetaObjectFactoryImpl::isCompileComplete()
{
    return obj_system.GetIsCompiledComplete();
}

bool MetaObjectFactoryImpl::swapObjects()
{
    if (isCompileComplete())
    {
        return obj_system.LoadCompiledModule();
    }
    return false;
}

void MetaObjectFactoryImpl::setCompileCallback(std::function<void(const std::string, int)>&)
{
}

std::shared_ptr<Connection> MetaObjectFactoryImpl::connectConstructorAdded(TSlot<void(void)>* slot)
{
    return on_constructor_added.connect(slot);
}

ObjectConstructor<MetaObjectFactoryImpl>::ObjectConstructor(SystemTable* table_)
    : table(table_)
{
}

std::shared_ptr<MetaObjectFactoryImpl> ObjectConstructor<MetaObjectFactoryImpl>::createShared() const
{
    return SharedPtr_t(new MetaObjectFactoryImpl(table));
}

std::unique_ptr<MetaObjectFactoryImpl> ObjectConstructor<MetaObjectFactoryImpl>::createUnique() const
{
    return UniquePtr_t(new MetaObjectFactoryImpl(table));
}

MetaObjectFactory* ObjectConstructor<MetaObjectFactoryImpl>::create() const
{
    return new MetaObjectFactoryImpl(table);
}
