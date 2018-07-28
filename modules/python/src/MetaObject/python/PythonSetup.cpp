#include "ct/reflect.hpp"
#include <opencv2/core/types.hpp>

#include "DataConverter.hpp"
#include "MetaObject.hpp"
#include "MetaObject/core.hpp"
#include "MetaObject/core/SystemTable.hpp"
#include "MetaObject/object/IMetaObject.hpp"
#include "MetaObject/object/MetaObjectFactory.hpp"
#include "MetaObject/params/ParamFactory.hpp"
#include "PythonAllocator.hpp"
#include "PythonPolicy.hpp"
#include "PythonSetup.hpp"
#include <MetaObject/core/detail/opencv_allocator.hpp>
#include <MetaObject/logging/logging.hpp>
#include <MetaObject/logging/profiling.hpp>
#include <signal.h> // SIGINT, etc

#include <RuntimeObjectSystem/IRuntimeObjectSystem.h>
#include <RuntimeObjectSystem/InheritanceGraph.hpp>
#include <RuntimeObjectSystem/InterfaceDatabase.hpp>

#include <boost/log/expressions.hpp>
#include <boost/python.hpp>
#include <boost/python/default_call_policies.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/thread.hpp>
#include <vector>

namespace boost
{
    inline const mo::ParamBase* get_pointer(const mo::ParamBase* ptr) { return ptr; }

    inline const mo::ICoordinateSystem* get_pointer(const mo::ICoordinateSystem* cs) { return cs; }

    inline const mo::ICoordinateSystem* get_pointer(const std::shared_ptr<mo::ICoordinateSystem>& ptr)
    {
        return ptr.get();
    }
}

namespace mo
{
    void setupEnums(const std::string& module_name);

    void setupDataTypes(const std::string& module_name);

    void setupPlugins(const std::string& module_name);

    inline const mo::ParamBase* get_pointer(const mo::ParamBase* ptr) { return ptr; }

    inline const mo::ICoordinateSystem* get_pointer(const mo::ICoordinateSystem* cs) { return cs; }

    inline mo::ICoordinateSystem* get_pointer(const std::shared_ptr<mo::ICoordinateSystem>& ptr) { return ptr.get(); }

    std::vector<std::string> listConstructableObjects()
    {
        auto ctrs = mo::MetaObjectFactory::instance().getConstructors();
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

    std::vector<IObjectInfo*> listObjectInfos()
    {
        auto ctrs = mo::MetaObjectFactory::instance().getConstructors();
        std::vector<IObjectInfo*> output;
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
            if (level == "trace")
                boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::trace);
            if (level == "debug")
                boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::debug);
            if (level == "info")
                boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
            if (level == "warning")
                boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::warning);
            if (level == "error")
                boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::error);
            if (level == "fatal")
                boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::fatal);
        }

        static std::vector<std::function<void(void)>> setup_functions;

        std::map<uint32_t,
                 std::pair<std::function<void(void)>, std::function<void(std::vector<IObjectConstructor*>&)>>>&
        interfaceSetupFunctions()
        {
            static std::map<
                uint32_t,
                std::pair<std::function<void(void)>, std::function<void(std::vector<IObjectConstructor*>&)>>>
                data;
            return data;
        }
        static bool setup = false;
        static std::string module_name;

        std::string getModuleName() { return module_name; }

        void setModuleName(const std::string& name) { module_name = name; }

        void registerSetupFunction(std::function<void(void)>&& func)
        {
            if (!setup)
            {
                setup_functions.emplace_back(std::move(func));
            }
            else
            {
                func();
            }
        }

        void registerInterfaceSetupFunction(uint32_t interface_id,
                                            std::function<void(void)>&& interface_func,
                                            std::function<void(std::vector<IObjectConstructor*>&)>&& func)
        {
            auto& data = interfaceSetupFunctions();
            if (data.find(interface_id) == data.end())
            {
                data[interface_id] = {std::move(interface_func), std::move(func)};
            }
            else
            {
                MO_LOG(warning) << "Interface id " << interface_id << " already registered";
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
            auto system = mo::MetaObjectFactory::instance().getObjectSystem();
            auto ifaces = system->GetInterfaces();
            MO_LOG(debug) << "Creating inheritance graph for " << ifaces.size() << " interfaces";
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
            MO_LOG(info) << "Constructable objects to python";
            auto graph = createGraph();
            auto ctrs = mo::MetaObjectFactory::instance().getConstructors();
            auto funcs = interfaceSetupFunctions();
            std::map<uint32_t, std::function<void(std::vector<IObjectConstructor*>&)>> func_map;
            for (auto& func : funcs)
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
            MO_LOG(info) << "Registering interfaces to python";
            auto graph = createGraph();
            auto funcs = interfaceSetupFunctions();
            std::map<uint32_t, std::function<void()>> func_map;
            for (auto& func : funcs)
            {
                func_map[func.first] = func.second.first;
            }
            registerInterfacesHelper(graph, func_map);
        }
        void cvErrorHandler() {}

        void sig_handler(int s)
        {
            switch (s)
            {
            case SIGSEGV:
            {
                std::cout << "Caught SIGSEGV " << mo::printCallstack(2, true);
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
                : m_system_table(SystemTable::instance()), m_factory(m_system_table.get()),
                  m_allocator(mo::Allocator::createAllocator()), m_cpu_allocator(m_allocator),
                  m_gpu_allocator(m_allocator), m_numpy_allocator(&m_cpu_allocator)
            {
                m_factory.registerTranslationUnit();
                m_default_opencv_allocator = cv::Mat::getStdAllocator();
                m_default_opencv_gpu_allocator = cv::cuda::GpuMat::defaultAllocator();
                m_system_table->allocator = m_allocator;
                m_allocator->setDefaultAllocator(m_allocator);
                m_factory.registerTranslationUnit();
                auto ret = signal(SIGINT, &sig_handler);
                if (ret == SIG_ERR)
                {
                    MO_LOG(warning) << "Error setting signal handler for SIGINT";
                }
                ret = signal(SIGSEGV, &sig_handler);
                if (ret == SIG_ERR)
                {
                    MO_LOG(warning) << "Error setting signal handler for SIGSEGV";
                }
                int devices = cv::cuda::getCudaEnabledDeviceCount();
                MO_ASSERT(devices);
                cv::cuda::GpuMat mat(10, 10, CV_32F);
                cv::Mat::setDefaultAllocator(&m_numpy_allocator);
                cv::cuda::GpuMat::setDefaultAllocator(&m_gpu_allocator);
            }

            void setAllocator(AllocatorMode mode)
            {
                if (mode == DEFAULT)
                {
                    m_numpy_allocator.default_allocator = m_default_opencv_allocator;
                    cv::cuda::GpuMat::setDefaultAllocator(m_default_opencv_gpu_allocator);
                }
            }

            ~LibGuard() {}

            std::shared_ptr<SystemTable> m_system_table;
            mo::MetaObjectFactory m_factory;
            std::shared_ptr<mo::Allocator> m_allocator;
            CvAllocatorProxy<CPU> m_cpu_allocator;
            CvAllocatorProxy<GPU> m_gpu_allocator;
            mo::NumpyAllocator m_numpy_allocator;
            cv::MatAllocator* m_default_opencv_allocator = nullptr;
            cv::cuda::GpuMat::Allocator* m_default_opencv_gpu_allocator = nullptr;
        };

        std::shared_ptr<SystemTable> pythonSetup(const char* module_name_)
        {
            mo::initProfiling();
            boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
            std::string module_name(module_name_);
            mo::python::module_name = module_name;
            setupAllocator();
            boost::shared_ptr<LibGuard> lib_guard(new LibGuard());
            mo::initCoreModule(lib_guard->m_system_table.get());
            mo::setupEnums(module_name);
            setupDataTypes(module_name);
            boost::python::def("listConstructableObjects", &listConstructableObjects);
            boost::python::def("listObjectInfos", &listObjectInfos);
            boost::python::def("log", &setLogLevel);

            for (const auto& func : mo::python::setup_functions)
            {
                func();
            }
            // RegisterInterface<IMetaObject> metaobject(&mo::python::setupInterface, &mo::python::setupObjects);
            setupInterface();
            // registerInterfaceSetupFunction(
            //    IMetaObject::getHash(), &mo::python::setupInterface, &mo::python::setupObjects);
            setupPlugins(module_name);
            mo::python::setup = true;

            boost::python::enum_<LibGuard::AllocatorMode>("AllocatorMode")
                .value("Default", LibGuard::DEFAULT)
                .value("Managed", LibGuard::POOLED);

            boost::python::class_<LibGuard, boost::shared_ptr<LibGuard>, boost::noncopyable> libguardobj(
                "LibGuard", boost::python::no_init);
            libguardobj.def("setAllocator", &LibGuard::setAllocator);
            boost::python::scope().attr("__libguard") = lib_guard;
            return lib_guard->m_system_table;
        }
    }
}

BOOST_PYTHON_MODULE(metaobject)
{
    mo::python::pythonSetup("metaobject");
}
