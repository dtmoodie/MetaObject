#include "PythonSetup.hpp"
#include "DataConverter.hpp"
#include "MetaObject.hpp"
#include "Parameters.hpp"
#include "PythonAllocator.hpp"
#include "PythonPolicy.hpp"

#include "ct/reflect.hpp"
#include <ct/interop/boost_python/ReflectedConverter.hpp>

#include <opencv2/core/types.hpp>

#include <MetaObject/core.hpp>
#include <MetaObject/core/SystemTable.hpp>
#include <MetaObject/cuda/MemoryBlock.hpp>
#include <MetaObject/cuda/opencv.hpp>
#include <MetaObject/object/IMetaObject.hpp>
#include <MetaObject/object/MetaObjectFactory.hpp>
#include <MetaObject/params/ParamFactory.hpp>
#include <MetaObject/thread/Thread.hpp>
#include <MetaObject/thread/ThreadPool.hpp>

#include <MetaObject/core/detail/allocator_policies/Combined.hpp>
#include <MetaObject/core/detail/allocator_policies/Pool.hpp>
#include <MetaObject/core/detail/allocator_policies/Stack.hpp>
#include <MetaObject/core/detail/allocator_policies/opencv.hpp>

#include <MetaObject/logging/logging.hpp>
#include <MetaObject/logging/profiling.hpp>

#include <signal.h> // SIGINT, etc

#include <RuntimeObjectSystem/IRuntimeObjectSystem.h>
#include <RuntimeObjectSystem/InheritanceGraph.hpp>
#include <RuntimeObjectSystem/InterfaceDatabase.hpp>

#include <boost/python.hpp>
#include <boost/python/default_call_policies.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/stacktrace.hpp>
#include <boost/thread.hpp>

#include <vector>

namespace boost
{
    inline const mo::IParam* get_pointer(const mo::IParam* ptr)
    {
        return ptr;
    }
} // namespace boost

namespace
{
#ifndef _MSC_VER
#include <dlfcn.h>
    void loadMetaParams(SystemTable* table)
    {
        void* handle = dlopen("libmetaobject_metaparams.so", RTLD_NOW | RTLD_GLOBAL);
        if (nullptr == handle)
        {
            handle = dlopen("libmetaobject_metaparamsd.so", RTLD_NOW | RTLD_GLOBAL);
        }
        if (nullptr == handle)
        {
            MO_LOG(info, "Unable to load metaparams");
            return;
        }
        auto* init = reinterpret_cast<void (*)(SystemTable*)>(dlsym(handle, "initModuleWithSystemTable"));
        if (init)
        {
            init(table);
        }
    }
#else
    void loadMetaParams()
    {
    }
#endif
} // namespace

namespace mo
{
    void setupEnums(const std::string& module_name);

    void setupPlugins(const std::string& module_name);

    inline const mo::IParam* get_pointer(const mo::IParam* ptr)
    {
        return ptr;
    }

    inline const mo::ICoordinateSystem* get_pointer(const mo::ICoordinateSystem* cs)
    {
        return cs;
    }

    inline mo::ICoordinateSystem* get_pointer(const std::shared_ptr<mo::ICoordinateSystem>& ptr)
    {
        return ptr.get();
    }

    std::vector<std::string> listConstructableObjects()
    {
        auto ctrs = mo::MetaObjectFactory::instance()->getConstructors();
        std::vector<std::string> output;
        for (auto ctr : ctrs)
        {
            auto info = ctr->GetObjectInfo();
            if (info)
            {
                output.push_back(info->GetObjectName());
            }
        }
        return output;
    }

    std::vector<const IObjectInfo*> listObjectInfos()
    {
        auto ctrs = mo::MetaObjectFactory::instance()->getConstructors();
        std::vector<const IObjectInfo*> output;
        for (auto ctr : ctrs)
        {
            auto info = ctr->GetObjectInfo();
            if (info)
            {
                output.push_back(info);
            }
        }
        return output;
    }

    namespace python
    {

        void setLogLevel(const std::string& level)
        {
            spdlog::logger& logger = mo::getDefaultLogger();
            if (level == "trace")
            {
                logger.set_level(spdlog::level::trace);
            }
            else if (level == "debug")
            {
                logger.set_level(spdlog::level::debug);
            }
            else if (level == "info")
            {
                logger.set_level(spdlog::level::info);
            }
            else if (level == "warning" || level == "warn")
            {
                logger.set_level(spdlog::level::warn);
            }
            else if (level == "error")
            {
                logger.set_level(spdlog::level::err);
            }
            else if (level == "fatal")
            {
                logger.set_level(spdlog::level::critical);
            }
        }
        struct PythonSetup
        {
            using InterfaceSetup_t = std::function<void(void)>;
            using ObjectSetup_t = std::function<void(std::vector<IObjectConstructor*>&)>;

            std::vector<std::function<void(void)>> setup_functions;
            std::string module_name;
            std::map<uint32_t, std::pair<InterfaceSetup_t, ObjectSetup_t>> interface_setup_functions;

            bool setup = false;
        };

        std::string getModuleName()
        {
            return singleton<PythonSetup>()->module_name;
        }

        void setModuleName(const std::string& name)
        {
            singleton<PythonSetup>()->module_name = name;
        }

        void registerSetupFunction(SystemTable* table, std::function<void(void)>&& func)
        {
            auto inst = table->getSingleton<PythonSetup>();

            if (!inst->setup)
            {
                inst->setup_functions.emplace_back(std::move(func));
            }
            else
            {
                func();
            }
        }

        void registerInterfaceSetupFunction(SystemTable* table,
                                            uint32_t interface_id,
                                            std::function<void(void)>&& interface_func,
                                            std::function<void(std::vector<IObjectConstructor*>&)>&& func)
        {
            auto& data = table->getSingleton<PythonSetup>()->interface_setup_functions;
            if (data.find(interface_id) == data.end())
            {
                data[interface_id] = {std::move(interface_func), std::move(func)};
            }
            else
            {
                MO_LOG(warn, "Interface id {} already registered", interface_id);
            }
        }

        void registerObjectsHelper(
            rcc::InheritanceGraph& graph,
            std::vector<IObjectConstructor*>& ctrs,
            std::map<uint32_t, std::function<void(std::vector<IObjectConstructor*>&)>> setup_functions)
        {
            for (auto itr = graph.interfaces.begin(); itr != graph.interfaces.end();)
            {
                if (itr->second.children.empty())
                {
                    auto func_itr = setup_functions.find(itr->first);
                    for (unsigned int iid : itr->second.parents)
                    {
                        graph.interfaces[iid].children.erase(itr->first);
                    }
                    if (func_itr != setup_functions.end())
                    {
                        func_itr->second(ctrs);
                    }
                    itr = graph.interfaces.erase(itr);
                }
                else
                {
                    ++itr;
                }
            }
            if (!graph.interfaces.empty())
            {
                registerObjectsHelper(graph, ctrs, setup_functions);
            }
        }

        rcc::InheritanceGraph createGraph()
        {
            rcc::InheritanceGraph graph;
            auto system = mo::MetaObjectFactory::instance()->getObjectSystem();
            auto ifaces = system->GetInterfaces();
            MO_LOG(debug, "Creating inheritance graph for {} interfaces", ifaces.size());
            // graph.interfaces.resize(ifaces.size());
            size_t i = 0;
            for (auto& info : ifaces)
            {
                graph.interfaces[info.iid].name = info.name;
                ++i;
            }

            for (i = 0; i < ifaces.size(); ++i)
            {
                auto& info = ifaces[i];
                for (size_t j = 0; j < ifaces.size(); ++j)
                {
                    if (i != j)
                    {
                        if (info.direct_inheritance_f(ifaces[j].iid))
                        {
                            graph.interfaces[ifaces[i].iid].parents.insert(ifaces[j].iid);
                            graph.interfaces[ifaces[j].iid].children.insert(ifaces[i].iid);
                        }
                    }
                }
            }
            return graph;
        }

        void registerObjects()
        {
            MO_LOG(info, "Constructable objects to python");
            auto graph = createGraph();
            auto ctrs = mo::MetaObjectFactory::instance()->getConstructors();
            auto setup = singleton<PythonSetup>();
            std::map<uint32_t, std::function<void(std::vector<IObjectConstructor*>&)>> func_map;
            for (auto& func : setup->interface_setup_functions)
            {
                func_map[func.first] = func.second.second;
            }
            registerObjectsHelper(graph, ctrs, func_map);
        }

        void registerInterfacesHelper(rcc::InheritanceGraph& graph,
                                      std::map<uint32_t, std::function<void()>> setup_functions)
        {
            for (auto itr = graph.interfaces.begin(); itr != graph.interfaces.end();)
            {
                if (itr->second.parents.empty())
                {
                    auto func_itr = setup_functions.find(itr->first);
                    for (unsigned int iid : itr->second.children)
                    {
                        graph.interfaces[iid].parents.erase(itr->first);
                    }
                    if (func_itr != setup_functions.end())
                    {
                        func_itr->second();
                    }
                    itr = graph.interfaces.erase(itr);
                }
                else
                {
                    ++itr;
                }
            }
            if (!graph.interfaces.empty())
            {
                registerInterfacesHelper(graph, setup_functions);
            }
        }

        void registerInterfaces()
        {
            MO_LOG(info, "Registering interfaces to python");
            auto graph = createGraph();
            auto setup = singleton<PythonSetup>();
            std::map<uint32_t, std::function<void()>> func_map;
            for (auto& func : setup->interface_setup_functions)
            {
                func_map[func.first] = func.second.first;
            }
            registerInterfacesHelper(graph, func_map);
        }
        void cvErrorHandler()
        {
        }

        void sig_handler(int s)
        {
            switch (s)
            {
            case SIGSEGV:
            {
                boost::stacktrace::stacktrace st;

                std::cout << "Caught SIGSEGV " << st;
                std::terminate();
                break;
            }
            case SIGINT:
            {
                std::cout << "Caught SIGINT, shutting down" << std::endl;
                static int count = 0;
                ++count;
                if (count > 2)
                {
                    std::terminate();
                }
                return;
            }
            case SIGILL:
            {
                std::cout << "Caught SIGILL " << std::endl;
                break;
            }
            case SIGTERM:
            {
                std::cout << "Caught SIGTERM " << std::endl;
                break;
            }
#ifndef _MSC_VER
            case SIGKILL:
            {
                std::cout << "Caught SIGKILL " << std::endl;
                break;
            }
#endif
            default:
            {
                std::cout << "Caught signal " << s << std::endl;
            }
            }
        }

        struct LibGuard
        {
            enum AllocatorMode
            {
                DEFAULT,
                POOLED
            };

            LibGuard()
            {
                m_system_table = SystemTable::instance();
                m_factory = mo::MetaObjectFactory::instance();
                m_factory->registerTranslationUnit();

                m_default_opencv_allocator = cv::Mat::getStdAllocator();
                m_default_opencv_gpu_allocator = cv::cuda::GpuMat::defaultAllocator();

                {
                    using Pool_t = mo::PoolPolicy<cuda::HOST>;
                    using Stack_t = mo::StackPolicy<cuda::HOST>;
                    using Allocator_t = mo::CombinedPolicy<Pool_t, Stack_t>;
                    m_host_allocator = std::make_shared<Allocator_t>();
                    m_cv_cpu_allocator.reset(new mo::CvAllocatorProxy(m_host_allocator.get()));
                }
                {
                    using Pool_t = mo::PoolPolicy<cuda::CUDA>;
                    using Stack_t = mo::StackPolicy<cuda::CUDA>;
                    using Allocator_t = mo::CombinedPolicy<Pool_t, Stack_t>;
                    m_device_allocator = std::make_shared<Allocator_t>();
                    m_cv_gpu_allocator.reset(new mo::cuda::AllocatorProxy<>(m_device_allocator.get()));
                }

                m_factory->registerTranslationUnit();
                auto ret = signal(SIGINT, &sig_handler);
                if (ret == SIG_ERR)
                {
                    MO_LOG(warn, "Error setting signal handler for SIGINT");
                }
                ret = signal(SIGSEGV, &sig_handler);
                if (ret == SIG_ERR)
                {
                    MO_LOG(warn, "Error setting signal handler for SIGSEGV");
                }
                const auto num_cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
                MO_ASSERT(num_cuda_devices > 0);
                cv::Mat::setDefaultAllocator(&m_numpy_allocator);
                cv::cuda::GpuMat::setDefaultAllocator(m_cv_gpu_allocator.get());
                m_callback_registry = python::ParamCallbackContainer::registry();
            }

            void setAllocator(AllocatorMode mode)
            {
                if (mode == DEFAULT)
                {
                    m_numpy_allocator.default_allocator = m_default_opencv_allocator;
                    cv::cuda::GpuMat::setDefaultAllocator(m_default_opencv_gpu_allocator);
                }
            }

            std::shared_ptr<SystemTable> m_system_table;
            std::shared_ptr<mo::MetaObjectFactory> m_factory;

            std::shared_ptr<mo::Allocator> m_host_allocator;
            std::shared_ptr<mo::DeviceAllocator> m_device_allocator;

            std::unique_ptr<mo::CvAllocatorProxy> m_cv_cpu_allocator;
            std::unique_ptr<mo::cuda::AllocatorProxy<>> m_cv_gpu_allocator;

            mo::NumpyAllocator m_numpy_allocator;
            cv::MatAllocator* m_default_opencv_allocator = nullptr;
            cv::cuda::GpuMat::Allocator* m_default_opencv_gpu_allocator = nullptr;
            std::shared_ptr<python::ParamCallbackContainer::Registry_t> m_callback_registry;
        };

        std::shared_ptr<mo::Thread> createThread()
        {
            auto pool = SystemTable::instance()->getSingleton<mo::ThreadPool>();
            return pool->requestThread();
        };

        void eventLoop(int milliseconds)
        {
            static mo::ConditionVariable cv;
            static mo::Mutex_t mtx;
            mo::Mutex_t::Lock_t lock(mtx);
            cv.wait_for(lock, std::chrono::milliseconds(milliseconds));
        }

        std::shared_ptr<SystemTable> pythonSetup(const char* module_name_)
        {
            std::string module_name(module_name_);
            auto table = SystemTable::instance();
            auto setup = table->getSingleton<PythonSetup>();
            setup->module_name = module_name;
            setupAllocator();
            boost::shared_ptr<LibGuard> lib_guard(new LibGuard());
            mo::initProfiling();
            mo::initCoreModule(lib_guard->m_system_table.get());
            mo::setupEnums(module_name);
            setupParameters(module_name);
            boost::python::def("listConstructableObjects", &listConstructableObjects);
            boost::python::def("listObjectInfos", &listObjectInfos);
            boost::python::def("log", &setLogLevel);

            static_assert(ct::ReflectImpl<mo::Thread>::NUM_FIELDS > 0, "");
            static_assert(ct::IsReflected<mo::Thread>::value, "");

            boost::python::class_<mo::Thread, std::shared_ptr<mo::Thread>, boost::noncopyable> thread(
                "Thread", boost::python::no_init);
            ct::detail::addProperties<mo::Thread>(thread);
            thread.def("__getitem__", &ct::detail::getItem<mo::Thread>);
            thread.def("__setitem__", &ct::detail::setItem<mo::Thread>);
            thread.def("__repr__", &ct::detail::repr<mo::Thread>);
            thread.def("__len__", &ct::detail::len<mo::Thread>);

            boost::python::def("createThread", &createThread);

            boost::python::class_<mo::IAsyncStream, std::shared_ptr<mo::IAsyncStream>, boost::noncopyable> stream(
                "Stream", boost::python::no_init);
            ct::detail::addProperties<mo::IAsyncStream>(stream);

            boost::python::enum_<mo::PriorityLevels>("PriorityLevels")
                .value("NONE", mo::PriorityLevels::NONE)
                .value("LOWEST", mo::PriorityLevels::LOWEST)
                .value("LOW", mo::PriorityLevels::LOWEST)
                .value("MEDIUM", mo::PriorityLevels::MEDIUM)
                .value("HIGH", mo::PriorityLevels::HIGH)
                .value("HIGHEST", mo::PriorityLevels::HIGHEST);

            boost::python::def("createStream",
                               &mo::IAsyncStream::create,
                               (boost::python::arg("name") = "",
                                boost::python::arg("device_id") = 0,
                                boost::python::arg("device_priority") = mo::MEDIUM,
                                boost::python::arg("thread_priority") = mo::MEDIUM));

            boost::python::def("eventLoop", &eventLoop);

            loadMetaParams(lib_guard->m_system_table.get());

            for (const auto& func : setup->setup_functions)
            {
                func();
            }
            setupInterface();
            setupPlugins(module_name);
            setup->setup = true;

            boost::python::enum_<LibGuard::AllocatorMode>("AllocatorMode")
                .value("Default", LibGuard::DEFAULT)
                .value("Managed", LibGuard::POOLED);

            boost::python::class_<LibGuard, boost::shared_ptr<LibGuard>, boost::noncopyable> libguardobj(
                "LibGuard", boost::python::no_init);
            libguardobj.def("setAllocator", &LibGuard::setAllocator);
            boost::python::scope().attr("__libguard") = lib_guard;

            return lib_guard->m_system_table;
        }
    } // namespace python
} // namespace mo

BOOST_PYTHON_MODULE(metaobject)
{
    mo::python::pythonSetup("metaobject");
}
