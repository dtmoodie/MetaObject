#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Parameters.hpp"
#include "MetaObject/core/TypeTable.hpp"
#include "MetaObject/object/IMetaObject.hpp"
#include "MetaObject/params/IParam.hpp"
#include "MetaObject/params/ISubscriber.hpp"
#include "MetaObject/python/DataConverter.hpp"
#include "PythonAllocator.hpp"
#include <boost/python.hpp>

#include <RuntimeObjectSystem/IObjectInfo.h>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

namespace mo
{
    std::string printParam(const IParam* param)
    {
        std::stringstream ss;
        param->print(ss);
        return ss.str();
    }

    std::string printInputParam(const mo::ISubscriber* param)
    {
        std::stringstream ss;
        ss << printParam(param);
        ss << "\n";
        auto input = param->getPublisher();
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

    boost::python::object getData(const IParam* param)
    {
        if (param->checkFlags(ParamFlags::kCONTROL))
        {
            auto control = static_cast<const IControlParam*>(param);
            if (control)
            {
                auto getter = mo::python::DataConverterRegistry::instance()->getGetter(control->getTypeInfo());
                if (getter)
                {
                    return getter(control);
                }
                MO_LOG(trace,
                       "Accessor function not found for for {} Available converters:\n {}",
                       control->getTypeInfo(),
                       printTypes());
                return boost::python::object(std::string("No to python converter registered for ") +
                                             TypeTable::instance()->typeToName(control->getTypeInfo()));
            }
        }
        return boost::python::object();
    }

    bool setData(mo::IParam* param, const boost::python::object& obj)
    {
        if (param->checkFlags(ParamFlags::kCONTROL))
        {
            auto control = static_cast<IControlParam*>(param);
            if (control)
            {
                auto setter = mo::python::DataConverterRegistry::instance()->getSetter(control->getTypeInfo());
                if (setter)
                {
                    return setter(control, obj);
                }
                MO_LOG(trace, "Setter function not found for {}", control->getTypeInfo());
            }
        }
        return false;
    }

    std::string getDataTypeName(const mo::IParam* param)
    {
        if (param->checkFlags(ParamFlags::kCONTROL))
        {
            auto control = static_cast<const IControlParam*>(param);
            if (control)
            {
                return mo::TypeTable::instance()->typeToName(control->getTypeInfo());
            }
        }
        return {};
    }

    python::ParamCallbackContainer::ParamCallbackContainer(mo::IControlParam* ptr, const boost::python::object& obj)
        : m_ptr(ptr)
        , m_callback(obj)
    {
        m_slot.bind(&ParamCallbackContainer::onParamUpdate, this);
        m_delete_slot.bind(&ParamCallbackContainer::onParamDelete, this);
        if (ptr)
        {
            m_getter = mo::python::DataConverterRegistry::instance()->getGetter(ptr->getTypeInfo());
            update_connection = ptr->registerUpdateNotifier(m_slot);
            del_connection = ptr->registerDeleteNotifier(m_delete_slot);
        }
    }

    void
    python::ParamCallbackContainer::onParamUpdate(const IParam& param, Header header, UpdateFlags flags, IAsyncStream&)
    {
        // TODO use stream?
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
            m_getter = mo::python::DataConverterRegistry::instance()->getGetter(m_ptr->getTypeInfo());
        }
        auto obj = m_getter(m_ptr);
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

    void python::ParamCallbackContainer::onParamDelete(const IParam&)
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
        if (param->checkFlags(ParamFlags::kCONTROL))
        {
            if (auto control = static_cast<IControlParam*>(param))
            {
                auto registry = python::ParamCallbackContainer::registry();
                (*registry)[param].emplace_back(
                    python::ParamCallbackContainer::Ptr_t(new python::ParamCallbackContainer(control, obj)));
            }
        }
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

        boost::python::class_<IParam, boost::noncopyable>("IParam", boost::python::no_init)
            .def("setCallback", &addCallback)
            .def("getName", &IParam::getTreeName)
            .def("getType", &getDataTypeName)
            .def("__repr__", &printParam)
            .add_property("data", &getData, &setData);

        boost::python::class_<std::vector<IParam*>> iparam_vec("IParamVec", boost::python::no_init);
        iparam_vec.def(boost::python::vector_indexing_suite<std::vector<IParam*>>());

        boost::python::class_<ISubscriber, ISubscriber*, boost::python::bases<IParam>, boost::noncopyable> input_param(
            "InputParam", boost::python::no_init);
        input_param.def("__repr__", &printInputParam);

        boost::python::implicitly_convertible<ISubscriber*, IParam*>();

        boost::python::class_<std::vector<ISubscriber*>> input_param_vec("InputParamVec", boost::python::no_init);
        input_param_vec.def(boost::python::vector_indexing_suite<std::vector<ISubscriber*>>());

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
} // namespace mo
