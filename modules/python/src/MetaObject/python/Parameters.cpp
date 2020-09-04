#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Parameters.hpp"
#include "MetaObject/core/TypeTable.hpp"
#include "MetaObject/object/IMetaObject.hpp"
#include "MetaObject/params/IParam.hpp"
#include "MetaObject/params/ISubscriber.hpp"
#include "MetaObject/python/DataConverter.hpp"

#include <MetaObject/runtime_reflection/DynamicVisitor.hpp>

#include "PythonAllocator.hpp"

#include <boost/python.hpp>

#include <RuntimeObjectSystem/IObjectInfo.h>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

namespace mo
{

    struct ToPythonVisitor : SaveCache
    {

        VisitorTraits traits() const override
        {
        }
        std::shared_ptr<Allocator> getAllocator() const override
        {
        }
        void setAllocator(std::shared_ptr<Allocator>) override
        {
        }
        ISaveVisitor& operator()(const bool* val, const std::string& name = "", size_t cnt = 1) override
        {
        }
        ISaveVisitor& operator()(const char* val, const std::string& name = "", size_t cnt = 1) override
        {
        }
        ISaveVisitor& operator()(const int8_t* val, const std::string& name = "", size_t cnt = 1) override
        {
        }
        ISaveVisitor& operator()(const uint8_t* val, const std::string& name = "", size_t cnt = 1) override
        {
        }
        ISaveVisitor& operator()(const int16_t* val, const std::string& name = "", size_t cnt = 1) override
        {
        }
        ISaveVisitor& operator()(const uint16_t* val, const std::string& name = "", size_t cnt = 1) override
        {
        }
        ISaveVisitor& operator()(const int32_t* val, const std::string& name = "", size_t cnt = 1) override
        {
        }
        ISaveVisitor& operator()(const uint32_t* val, const std::string& name = "", size_t cnt = 1) override
        {
        }
        ISaveVisitor& operator()(const int64_t* val, const std::string& name = "", size_t cnt = 1) override
        {
        }
        ISaveVisitor& operator()(const uint64_t* val, const std::string& name = "", size_t cnt = 1) override
        {
        }
#ifdef ENVIRONMENT64
#ifndef _MSC_VER
        ISaveVisitor& operator()(const long long* val, const std::string& name = "", size_t cnt = 1) override
        {
        }
        ISaveVisitor& operator()(const unsigned long long* val, const std::string& name = "", size_t cnt = 1) override
        {
        }
#endif
#else
        ISaveVisitor& operator()(const long int* val, const std::string& name = "", size_t cnt = 1) override
        {
        }
        ISaveVisitor& operator()(const unsigned long int* val, const std::string& name = "", size_t cnt = 1) override
        {
        }
#endif
        ISaveVisitor& operator()(const float* val, const std::string& name = "", size_t cnt = 1) override
        {
        }
        ISaveVisitor& operator()(const double* val, const std::string& name = "", size_t cnt = 1) override
        {
        }
        ISaveVisitor& operator()(const void* binary, const std::string& name = "", size_t bytes = 1) override
        {
        }

        ISaveVisitor&
        operator()(IStructTraits* trait, const void* inst, const std::string& name = "", size_t cnt = 1) override
        {
        }
        ISaveVisitor&
        operator()(IContainerTraits* trait, const void* inst, const std::string& name = "", size_t cnt = 1) override
        {
        }

        boost::python::object getObject()
        {
            return std::move(m_object);
        }

      private:
        boost::python::object m_object;
    };

    std::string printParam(const IParam* param)
    {
        std::stringstream ss;
        param->print(ss);
        return ss.str();
    }

    std::string printSubscriber(const mo::ISubscriber* param)
    {
        std::stringstream ss;
        param->print(ss);
        ss << "\n";
        auto input = param->getPublisher();
        if (input)
        {
            ss << "Input Publisher: " << printParam(input);
        }
        else
        {
            ss << "Input not set";
        }
        return ss.str();
    }

    std::string printPublisher(const mo::IPublisher* param)
    {
        std::stringstream ss;
        param->print(ss);
        ss << "\n Headers: ";
        auto headers = param->getAvailableHeaders();
        ss << headers;
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

    boost::python::object getData(IParam* param)
    {
        if (param->checkFlags(ParamFlags::kCONTROL))
        {
            auto control = static_cast<const IControlParam*>(param);
            if (control)
            {
                const mo::TypeInfo type = control->getTypeInfo();
                const mo::python::DataConverterRegistry* converter_registry =
                    mo::python::DataConverterRegistry::instance();
                auto getter = converter_registry->getGetter(type);
                if (getter)
                {
                    return getter(control);
                }
                MO_LOG(trace, "Accessor function not found for for {} Available converters:\n {}", type, printTypes());
                return boost::python::object(std::string("No to python converter registered for ") +
                                             TypeTable::instance()->typeToName(control->getTypeInfo()));
            }
        }
        if (param->checkFlags(ParamFlags::kINPUT))
        {
            const ISubscriber* sub = dynamic_cast<const ISubscriber*>(param);
            if (sub)
            {
                IDataContainerConstPtr_t data = sub->getCurrentData();
                if (data)
                {
                    ToPythonVisitor visitor;
                    data->save(visitor, "data_container");
                    return visitor.getObject();
                }
            }
        }
        if (param->checkFlags(ParamFlags::kOUTPUT))
        {
            IPublisher* pub = dynamic_cast<IPublisher*>(param);
            if (pub)
            {
                IDataContainerConstPtr_t data = pub->getData();
                if (data)
                {
                    ToPythonVisitor visitor;
                    data->save(visitor, "data_container");
                    return visitor.getObject();
                }
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

        boost::python::class_<ISubscriber, ISubscriber*, boost::python::bases<IParam>, boost::noncopyable> subscriber(
            "InputParam", boost::python::no_init);
        subscriber.def("__repr__", &printSubscriber);

        boost::python::class_<IPublisher, IPublisher*, boost::python::bases<IParam>, boost::noncopyable> publisher(
            "InputParam", boost::python::no_init);
        publisher.def("__repr__", &printPublisher);

        boost::python::class_<IControlParam, IControlParam*, boost::python::bases<IParam>, boost::noncopyable> control(
            "IControlParam", boost::python::no_init);

        boost::python::implicitly_convertible<ISubscriber*, IParam*>();

        boost::python::class_<std::vector<ISubscriber*>> input_param_vec("InputParamVec", boost::python::no_init);
        input_param_vec.def(boost::python::vector_indexing_suite<std::vector<ISubscriber*>>());

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
