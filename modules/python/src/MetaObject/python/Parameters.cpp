#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Parameters.hpp"
#include "MetaObject/core/TypeTable.hpp"
#include "MetaObject/core/TypeTable.hpp"
#include "MetaObject/object/IMetaObject.hpp"
#include "MetaObject/params/ICoordinateSystem.hpp"
#include "MetaObject/params/IParam.hpp"
#include "MetaObject/params/InputParam.hpp"
#include "MetaObject/python/DataConverter.hpp"
#include "PythonAllocator.hpp"
#include <boost/python.hpp>

#include <RuntimeObjectSystem/IObjectInfo.h>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

namespace mo
{
    std::string printParam(const mo::ParamBase* param)
    {
        std::stringstream ss;
        param->print(ss);
        return ss.str();
    }

    std::string printInputParam(const mo::InputParam* param)
    {
        std::stringstream ss;
        ss << printParam(param);
        ss << "\n";
        auto input = param->getInputParam();
        if (input)
        {
            ss << printParam(input);
        }
        return ss.str();
    }

    std::string printStringVec(const std::vector<std::string>& strs)
    {
        std::stringstream ss;
        for (const auto& itr : strs)
        {
            ss << itr << "\n";
        }
        return ss.str();
    }

    std::string printTypes()
    {
        auto types = mo::python::DataConverterRegistry::instance()->listConverters();
        std::stringstream ss;
        for (auto& type : types)
        {
            ss << mo::TypeTable::instance()->typeToName(type) << "\n";
        }
        return ss.str();
    }

    boost::python::object getData(const mo::ParamBase* param)
    {
        auto getter = mo::python::DataConverterRegistry::instance()->getGetter(param->getTypeInfo());
        if (getter)
        {
            return getter(param);
        }
        MO_LOG(trace,
               "Accessor function not found for for {} Available converters:\n {}",
               mo::TypeTable::instance()->typeToName(param->getTypeInfo()),
               printTypes());
        return boost::python::object(std::string("No to python converter registered for " +
                                                 mo::TypeTable::instance()->typeToName(param->getTypeInfo())));
    }

    bool setData(mo::IParam* param, const boost::python::object& obj)
    {
        auto setter = mo::python::DataConverterRegistry::instance()->getSetter(param->getTypeInfo());
        if (setter)
        {
            return setter(param, obj);
        }
        MO_LOG(trace, "Setter function not found for {}", mo::TypeTable::instance()->typeToName(param->getTypeInfo()));
        return false;
    }

    std::string getDataTypeName(const mo::ParamBase* param)
    {
        return mo::TypeTable::instance()->typeToName(param->getTypeInfo());
    }

    python::ParamCallbackContainer::ParamCallbackContainer(mo::IParam* ptr, const boost::python::object& obj)
        : m_ptr(ptr)
        , m_callback(obj)
    {
        m_slot = std::bind(&ParamCallbackContainer::onParamUpdate,
                           this,
                           std::placeholders::_1,
                           std::placeholders::_2,
                           std::placeholders::_3);
        m_delete_slot = std::bind(&ParamCallbackContainer::onParamDelete, this, std::placeholders::_1);
        if (ptr)
        {
            m_getter = mo::python::DataConverterRegistry::instance()->getGetter(ptr->getTypeInfo());
            update_connection = ptr->registerUpdateNotifier(&m_slot);
            del_connection = ptr->registerDeleteNotifier(&m_delete_slot);
        }
    }

    void python::ParamCallbackContainer::onParamUpdate(IParam* param, Header header, UpdateFlags flags)
    {
        if (m_ptr == nullptr)
        {
            return;
        }
        ct::PyEnsureGIL lock;
        if (!m_callback)
        {
            return;
        }
        if (!m_getter)
        {
            m_getter = mo::python::DataConverterRegistry::instance()->getGetter(param->getTypeInfo());
        }
        auto obj = m_getter(param);
        if (obj)
        {
            try
            {
                if (header.timestamp)
                {
                    m_callback(obj, header.timestamp->time_since_epoch().count(), header.frame_number, flags);
                }
                else
                {
                    m_callback(obj, boost::python::object(), header.frame_number, flags);
                }
            }
            catch (...)
            {
                MO_LOG(warn, "Callback invokation failed");
            }

            // m_callback();
        }
    }

    void python::ParamCallbackContainer::onParamDelete(const mo::ParamBase*)
    {
        m_ptr = nullptr;
    }

    auto python::ParamCallbackContainer::registry() -> std::shared_ptr<Registry_t>
    {
        static std::weak_ptr<Registry_t>* g_registry = nullptr;
        std::shared_ptr<Registry_t> out;
        if (g_registry == nullptr)
        {
            g_registry = new std::weak_ptr<Registry_t>();
            out = std::make_shared<Registry_t>();
            *g_registry = out;
        }
        else
        {
            out = g_registry->lock();
        }
        return out;
    }

    void addCallback(mo::IParam* param, const boost::python::object& obj)
    {
        auto registry = python::ParamCallbackContainer::registry();
        (*registry)[param].emplace_back(
            python::ParamCallbackContainer::Ptr_t(new python::ParamCallbackContainer(param, obj)));
    }

    std::string printTime(const mo::Time& ts)
    {
        std::stringstream ss;
        ss << ts;
        return ss.str();
    }

    const mo::Time& getTime(const mo::OptionalTime& ts)
    {
        return ts.get();
    }

    void python::setupParameters(const std::string& module_name)
    {
        boost::python::object datatype_module(
            boost::python::handle<>(boost::python::borrowed(PyImport_AddModule((module_name + ".datatypes").c_str()))));
        boost::python::scope().attr("datatypes") = datatype_module;
        boost::python::scope datatype_scope = datatype_module;

        boost::python::class_<ParamBase, boost::noncopyable>("ParamBase", boost::python::no_init)
            .def("getName", &ParamBase::getTreeName)
            .def("getType", &getDataTypeName)
            .def("__repr__", &printParam)
            .add_property("data", &getData, &setData);

        boost::python::class_<IParam, boost::noncopyable, boost::python::bases<ParamBase>>("IParam",
                                                                                           boost::python::no_init)
            .def("setCallback", &addCallback);

        boost::python::implicitly_convertible<IParam*, ParamBase*>();

        boost::python::class_<std::vector<ParamBase*>> param_vec("ParamVec", boost::python::no_init);
        param_vec.def(boost::python::vector_indexing_suite<std::vector<ParamBase*>>());

        boost::python::class_<std::vector<IParam*>> iparam_vec("IParamVec", boost::python::no_init);
        iparam_vec.def(boost::python::vector_indexing_suite<std::vector<IParam*>>());

        boost::python::class_<InputParam, InputParam*, boost::python::bases<ParamBase>, boost::noncopyable> input_param(
            "InputParam", boost::python::no_init);
        input_param.def("__repr__", &printInputParam);

        boost::python::implicitly_convertible<InputParam*, IParam*>();

        boost::python::class_<std::vector<InputParam*>> input_param_vec("InputParamVec", boost::python::no_init);
        input_param_vec.def(boost::python::vector_indexing_suite<std::vector<InputParam*>>());

        // boost::python::class_<ICoordinateSystem, std::shared_ptr<ICoordinateSystem>, boost::noncopyable>
        // cs_obj("ICoordinateSystem", boost::python::no_init);
        // cs_obj.def("getName", &ICoordinateSystem::getName,
        // boost::python::return_value_policy<boost::python::reference_existing_object>());

        boost::python::class_<IAsyncStream, std::shared_ptr<IAsyncStream>, boost::noncopyable>("AsyncStream",
                                                                                               boost::python::no_init)
            .add_property("name", &IAsyncStream::name)
            .add_property("thread_id", &IAsyncStream::threadId)
            .add_property("is_device", &IAsyncStream::isDeviceStream);

        boost::python::def("setDefaultBufferType", &setDefaultBufferType);

        boost::python::class_<IMetaObject, boost::noncopyable> metaobject("IMetaObject", boost::python::no_init);

        boost::python::class_<std::vector<std::string>>("StringVec")
            .def(boost::python::vector_indexing_suite<std::vector<std::string>>())
            .def("__repr__", &printStringVec);

        boost::python::class_<IObjectInfo, boost::noncopyable>("IObjectInfo", boost::python::no_init)
            .def("getInterfaceId", &IObjectInfo::GetInterfaceId)
            .def("getInterfaceName", &IObjectInfo::GetInterfaceName)
            .def("__repr__", &IObjectInfo::Print, (boost::python::arg("verbosity") = IObjectInfo::INFO));

        boost::python::class_<Time, boost::noncopyable> time("Time");
        // time.def("__repr__", &printTime);

        boost::python::class_<mo::OptionalTime, boost::noncopyable> optional_time("OptionalTime");
        // optional_time.def("get", &getTime);
    }
}
