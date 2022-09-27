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
#include <MetaObject/thread/ThreadInfo.hpp>

#include <boost/optional.hpp>

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

namespace std
{
    ostream& operator<<(ostream& os, const vector<IObjectConstructor*>& ctrs)
    {
        os << '[';
        for (size_t i = 0; i < ctrs.size(); ++i)
        {
            if (i != 0)
            {
                os << ' ';
            }
            os << ctrs[i]->GetName();
        }
        os << ']';
        return os;
    }
} // namespace std

class MetaObjectFactoryImpl : public MetaObjectFactory
{
  public:
    using Ptr_t = std::shared_ptr<MetaObjectFactory>;

    MetaObjectFactoryImpl(SystemTable* table)
        : m_logger_ptr(table->getLogger())
        , m_logger(*m_logger_ptr)
    {
        obj_system.Initialise(&m_compiler_logger, table);
#if !defined(NDEBUG) || defined(_DEBUG)
        obj_system.SetOptimizationLevel(RCppOptimizationLevel::RCCPPOPTIMIZATIONLEVEL_DEBUG);
#endif
    }

    ~MetaObjectFactoryImpl() override
    {
    }

    rcc::shared_ptr<IMetaObject> create(const char* type_name, int64_t interface_id = -1) override;

    rcc::shared_ptr<IMetaObject> get(ObjectId id, const char* type_name) override;

    std::vector<std::string> listConstructableObjects(int interface_id = -1) const override;
    std::string printAllObjectInfo(int64_t interface_id = -1) const override;

    std::vector<IObjectConstructor*> getConstructors(int64_t interface_id = -1) const override;
    IObjectConstructor* getConstructor(const char* type_name) const override;
    const IObjectInfo* getObjectInfo(const char* type_name) const override;
    std::vector<const IObjectInfo*> getAllObjectInfo() const override;

    bool loadPlugin(const std::string& filename) override;
    int loadPlugins(const std::string& path = "./") override;

    std::vector<std::string> listLoadedPlugins(PluginVerbosity verbosity = brief) const override;
    std::vector<PluginInfo> listLoadedPluginInfo() const override;

    // This function is inlined to guarantee it exists in the calling translation unit, which
    // thus makes certain to load the correct PerModuleInterface instance
    MO_INLINE void registerTranslationUnit();
    void setupObjectConstructors(IPerModuleInterface* pPerModuleInterface) override;
    IRuntimeObjectSystem* getObjectSystem() override;

    // Recompilation stuffs
    bool abortCompilation() override;
    bool checkCompile() override;
    bool isCurrentlyCompiling() override;
    bool isCompileComplete() override;
    bool swapObjects() override;
    void setCompileCallback(std::function<void(const std::string, int)>& f) override;
    std::shared_ptr<Connection> connectConstructorAdded(TSlot<void(void)>& slot) override;

    void setupPluginCompilationOptions(const int32_t project_id, PluginCompilationOptions options) override;

  private:
    RuntimeObjectSystem obj_system;
    CompileLogger m_compiler_logger;
    std::vector<PluginInfo> plugins;
    TSignal<void(void)> on_constructor_added;
    std::shared_ptr<spdlog::logger> m_logger_ptr;
    spdlog::logger& m_logger;
};

namespace mo
{
    template <>
    struct ObjectConstructor<MetaObjectFactoryImpl>
    {
        using SharedPtr_t = std::shared_ptr<MetaObjectFactoryImpl>;
        using UniquePtr_t = std::unique_ptr<MetaObjectFactoryImpl>;

        ObjectConstructor();

        SharedPtr_t createShared() const;

        UniquePtr_t createUnique() const;

        MetaObjectFactory* create() const;
    };
} // namespace mo

__attribute__((noinline)) MetaObjectFactory::Ptr_t MetaObjectFactory::instance(SystemTable* table)
{
    return table->getSingleton<MetaObjectFactory, MetaObjectFactoryImpl>();
}

uint32_t MetaObjectFactory::loadStandardPlugins()
{
    std::string postfix;
#ifdef _DEBUG
    postfix = "d";
#endif
#ifdef _MSC_VER
    return mo::MetaObjectFactory::instance()->loadPlugin("./bin/plugins");
#else
    return mo::MetaObjectFactory::instance()->loadPlugins("./bin/plugins");
#endif
}

MetaObjectFactory::~MetaObjectFactory()
{
}

IRuntimeObjectSystem* MetaObjectFactoryImpl::getObjectSystem()
{
    return &obj_system;
}

rcc::shared_ptr<IMetaObject> MetaObjectFactoryImpl::create(const char* type_name, int64_t interface_id)
{
    auto constructor = obj_system.GetObjectFactorySystem()->GetConstructor(type_name);
    if (constructor)
    {
        if (interface_id != -1)
        {
            if (constructor->GetInterfaceId() != interface_id)
            {
                return {};
            }
        }
        auto obj = constructor->Construct();
        rcc::shared_ptr<IMetaObject> mobj(obj);
        if (mobj)
        {
            mobj->Init(true);
            return mobj;
        }
    }
    auto ctrs = this->getConstructors(interface_id);
    m_logger.warn("No constructor found for {}.  Available constructors: \n{}", type_name, ctrs);
    return {};
}

rcc::shared_ptr<IMetaObject> MetaObjectFactoryImpl::get(ObjectId id, const char* type_name)
{
    AUDynArray<IObjectConstructor*> constructors;
    obj_system.GetObjectFactorySystem()->GetAll(constructors);
    if (id.m_ConstructorId < constructors.Size())
    {
        if (strcmp(constructors[id.m_ConstructorId]->GetName(), type_name) == 0)
        {
            auto obj = constructors[id.m_ConstructorId]->GetConstructedObject(id.m_PerTypeId);
            if (obj)
            {
                return rcc::shared_ptr<IMetaObject>(obj);
            }
            else
            {
                m_logger.debug("No object exits for {} with instance id of {}", type_name, id.m_PerTypeId);
            }
        }
        else
        {
            m_logger.debug("Requested type '{}' does not match constructor type '{}' for given ID",
                           type_name,
                           constructors[id.m_ConstructorId]->GetName());
            std::string str_type_name(type_name);
            for (size_t i = 0; i < constructors.Size(); ++i)
            {
                if (std::string(constructors[i]->GetName()) == str_type_name)
                {
                    auto obj = constructors[i]->GetConstructedObject(id.m_PerTypeId);
                    if (obj)
                    {
                        return rcc::shared_ptr<IMetaObject>(obj);
                    }
                    else
                    {
                        return {}; // Object just doesn't exist yet.
                    }
                }
            }
            m_logger.warn("Requested type {} not found", type_name);
        }
    }
    return {};
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

const IObjectInfo* MetaObjectFactoryImpl::getObjectInfo(const char* type_name) const
{
    auto constructor = obj_system.GetObjectFactorySystem()->GetConstructor(type_name);
    if (constructor)
    {
        return constructor->GetObjectInfo();
    }
    return nullptr;
}

std::vector<const IObjectInfo*> MetaObjectFactoryImpl::getAllObjectInfo() const
{
    std::vector<const IObjectInfo*> output;
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
            ss << plugins[i].m_path << " " << plugins[i].m_state << " (" << plugins[i].m_load_time << " ms)";
            break;
        case debug:
            ss << plugins[i].m_path << " " << plugins[i].m_state << " (" << plugins[i].m_load_time << " ms)";
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

bool MetaObjectFactoryImpl::loadPlugin(const std::string& full_plugin_path)
{
    static int projectCount = 0;
    m_logger.info("Loading plugin {}", full_plugin_path);

    if (!boost::filesystem::is_regular_file(full_plugin_path))
    {
        m_logger.warn("{} does not exist", boost::filesystem::absolute(full_plugin_path));
        return false;
    }
    PluginInfo plugin_info;
    std::string plugin_name = boost::filesystem::path(full_plugin_path).stem().string();
    plugin_info.m_path = full_plugin_path;
    const mo::Time start = mo::Time::now();
    HMODULE handle = LoadLibrary(full_plugin_path.c_str());
    if (handle == nullptr)
    {
        auto err = GetLastError();
        m_logger.warn("Failed to load {} dueo to '{} {}'", plugin_name, err, GetLastErrorAsString());
        plugin_info.m_state = "failed";
        plugins.push_back(plugin_info);
        return false;
    }
    typedef const char* (*InfoFunctor)();

    InfoFunctor info = (InfoFunctor)GetProcAddress(handle, "getPluginBuildInfo");
    if (info)
    {
        m_logger.debug(info());
        plugin_info.m_build_info = info();
    }

    typedef IPerModuleInterface* (*moduleFunctor)();
    moduleFunctor module = (moduleFunctor)GetProcAddress(handle, "GetPerModuleInterface");
    if (module)
    {
        auto moduleInterface = module();
        moduleInterface->SetModuleFileName(full_plugin_path.c_str());
        boost::filesystem::path path(full_plugin_path);
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
        mo::Time end = mo::Time::now();
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

template <class R>
boost::optional<R> invoke(void* handle, const char* name)
{
    typedef R (*Func)();
    auto func = (Func)(dlsym(handle, name));
    if (func)
    {
        return func();
    }
    return {};
}

bool MetaObjectFactoryImpl::loadPlugin(const std::string& full_plugin_path)
{
    std::string old_name = mo::getThisThreadName();
    m_logger.info("Loading {}", full_plugin_path);
    boost::filesystem::path path(full_plugin_path);
    if (!boost::filesystem::exists(path))
    {
        m_logger.warn("{} does not exist", boost::filesystem::absolute(full_plugin_path));
        return false;
    }
    std::string base = path.stem().replace_extension("").string();
    mo::setThisThreadName(base.substr(3));
    boost::posix_time::ptime start = boost::posix_time::microsec_clock::universal_time();
    void* handle = dlopen(full_plugin_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    // Fallback on old module
    if (handle == nullptr)
    {
        const char* dlsym_error = dlerror();
        if (dlsym_error)
        {
            m_logger.warn(dlsym_error);
            boost::posix_time::ptime end = boost::posix_time::microsec_clock::universal_time();
            plugins.emplace_back(full_plugin_path,
                                 "failed (dlsym_error - " + std::string(dlsym_error),
                                 (end - start).total_milliseconds(),
                                 nullptr);
            mo::setThisThreadName(old_name);
            return false;
        }
    }

    {
        typedef void (*InitFunctor)(SystemTable*);

        InitFunctor init = InitFunctor(dlsym(handle, "initModule"));
        if (init)
        {
            SystemTable* table = SystemTable::instance().get();
            init(table);
        }
    }

    auto pid = invoke<int>(handle, "getPluginProjectId");
    if (!pid || *pid == -1)
    {
        pid = obj_system.GetProjectCount();
    }

    auto info = invoke<const char*>(handle, "getPluginBuildInfo");
    if (info)
    {
        m_logger.debug(*info);
    }

    auto module = invoke<IPerModuleInterface*>(handle, "GetPerModuleInterface");
    if (!module)
    {
        const char* dlsym_error = dlerror();
        if (dlsym_error)
        {
            m_logger.warn(dlsym_error);
            boost::posix_time::ptime end = boost::posix_time::microsec_clock::universal_time();
            plugins.emplace_back(full_plugin_path,
                                 "failed (dlsym_error - " + std::string(dlsym_error),
                                 (end - start).total_milliseconds(),
                                 info ? *info : nullptr);
            mo::setThisThreadName(old_name);
            return false;
        }
        if (!module)
        {
            m_logger.warn("module == nullptr");
            boost::posix_time::ptime end = boost::posix_time::microsec_clock::universal_time();
            plugins.emplace_back(full_plugin_path,
                                 "failed (module == nullptr)",
                                 (end - start).total_milliseconds(),
                                 info ? *info : nullptr);
            mo::setThisThreadName(old_name);
        }
    }
    IPerModuleInterface* interface = *module;
    interface->SetModuleFileName(full_plugin_path.c_str());

    typedef void (*InitPlugin_f)(const int32_t, MetaObjectFactory*);
    InitPlugin_f init = InitPlugin_f(dlsym(handle, "initPlugin"));
    if (init)
    {
        init(*pid, this);
    }
    else
    {
        m_logger.debug("Loading plugin ({}) compile options:", full_plugin_path);

#ifdef _DEBUG
// obj_system.SetOptimizationLevel(RCCPPOPTIMIZATIONLEVEL_DEBUG, id);
#else
        obj_system.SetOptimizationLevel(RCCPPOPTIMIZATIONLEVEL_PERF, *pid);
#endif
        interface->SetProjectIdForAllConstructors(static_cast<unsigned short>(*pid));

        setupObjectConstructors(interface);
    }

    boost::posix_time::ptime end = boost::posix_time::microsec_clock::universal_time();
    plugins.emplace_back(
        full_plugin_path, "success", (end - start).total_milliseconds(), info ? *info : nullptr, pid ? *pid : 0);
    mo::setThisThreadName(old_name);
    return true;
}
#endif

template <class F>
void forEachString(const char** strings, const F& op)
{
    int32_t i = 0;
    while (strings[i] != nullptr)
    {
        op(strings[i]);
        ++i;
    }
}

void MetaObjectFactoryImpl::setupPluginCompilationOptions(const int32_t pid, PluginCompilationOptions options)
{
    forEachString(options.includes, [this, pid](const char* dir) {
        obj_system.AddIncludeDir(dir, pid);
        m_logger.debug("  -I{}", dir);
    });
#ifdef _DEBUG

    forEachString(options.link_dirs_debug, [this, pid](const char* dir) {
        obj_system.AddLibraryDir(dir, pid);
        m_logger.debug("-L{}", dir);
    });
#else
    forEachString(options.link_dirs_release, [this, pid](const char* dir) {
        obj_system.AddLibraryDir(dir, pid);
        m_logger.debug("-L{}", dir);
    });
#endif

    forEachString(options.compile_options, [this, pid](const char* opt) {
        obj_system.AppendAdditionalCompileOptions(opt, pid);
        m_logger.debug(opt);
    });

    forEachString(options.compile_definitions, [this, pid](const char* def) {
        obj_system.AppendAdditionalCompileOptions(def, pid);
        m_logger.debug("{}", def);
    });

    if (options.compiler != nullptr)
    {
        obj_system.SetCompilerLocation(options.compiler, pid);
        m_logger.debug("compiler location = {}", options.compiler);
    }
    forEachString(options.link_libs,
                  [this, pid](const char* lib) { obj_system.AppendAdditionalLinkLibraries(lib, pid); });
    forEachString(options.link_libs_debug,
                  [this, pid](const char* lib) { obj_system.AppendAdditionalDebugLinkLibraries(lib, pid); });

    forEachString(options.link_libs_release,
                  [this, pid](const char* lib) { obj_system.AppendAdditionalReleaseLinkLibraries(lib, pid); });
}

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

std::shared_ptr<Connection> MetaObjectFactoryImpl::connectConstructorAdded(TSlot<void(void)>& slot)
{
    return on_constructor_added.connect(slot);
}

ObjectConstructor<MetaObjectFactoryImpl>::ObjectConstructor()
{
}

std::shared_ptr<MetaObjectFactoryImpl> ObjectConstructor<MetaObjectFactoryImpl>::createShared() const
{
    auto module = PerModuleInterface::GetInstance();
    auto table = module->GetSystemTable();
    MO_ASSERT(table);
    return SharedPtr_t(new MetaObjectFactoryImpl(table));
}

std::unique_ptr<MetaObjectFactoryImpl> ObjectConstructor<MetaObjectFactoryImpl>::createUnique() const
{
    auto module = PerModuleInterface::GetInstance();
    auto table = module->GetSystemTable();
    MO_ASSERT(table);
    return UniquePtr_t(new MetaObjectFactoryImpl(table));
}

MetaObjectFactory* ObjectConstructor<MetaObjectFactoryImpl>::create() const
{
    auto module = PerModuleInterface::GetInstance();
    auto table = module->GetSystemTable();
    MO_ASSERT(table);
    return new MetaObjectFactoryImpl(table);
}
