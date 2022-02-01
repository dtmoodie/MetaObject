#ifndef MO_PYTHON_PYTHON_CONVERSION_VISITATION_HPP
#define MO_PYTHON_PYTHON_CONVERSION_VISITATION_HPP
#include "DataConverter.hpp"

#include <MetaObject/runtime_reflection/DynamicVisitor.hpp>

#include <boost/python.hpp>

namespace boost
{
    namespace python
    {
        bool hasattr(const object& o, const char* name);
        bool setattr(object& o, const object& v, const char* name);

        object dir(const object& o);
    } // namespace python
} // namespace boost

namespace mo
{
    namespace python
    {

        struct ParameterPythonWrapper
        {
        };

        struct DataConversionTable;
        struct ToPythonVisitor : SaveCache
        {
            ToPythonVisitor(const python::DataConversionTable* = python::DataConversionTable::instance());

            VisitorTraits traits() const override;
            std::shared_ptr<Allocator> getAllocator() const override;
            void setAllocator(std::shared_ptr<Allocator>) override;
            ISaveVisitor& operator()(const bool* val, const std::string& name = "", size_t cnt = 1) override;
            ISaveVisitor& operator()(const char* val, const std::string& name = "", size_t cnt = 1) override;
            ISaveVisitor& operator()(const int8_t* val, const std::string& name = "", size_t cnt = 1) override;
            ISaveVisitor& operator()(const uint8_t* val, const std::string& name = "", size_t cnt = 1) override;
            ISaveVisitor& operator()(const int16_t* val, const std::string& name = "", size_t cnt = 1) override;
            ISaveVisitor& operator()(const uint16_t* val, const std::string& name = "", size_t cnt = 1) override;
            ISaveVisitor& operator()(const int32_t* val, const std::string& name = "", size_t cnt = 1) override;
            ISaveVisitor& operator()(const uint32_t* val, const std::string& name = "", size_t cnt = 1) override;
            ISaveVisitor& operator()(const int64_t* val, const std::string& name = "", size_t cnt = 1) override;
            ISaveVisitor& operator()(const uint64_t* val, const std::string& name = "", size_t cnt = 1) override;
#ifdef ENVIRONMENT64
#ifndef _MSC_VER
            ISaveVisitor& operator()(const long long* val, const std::string& name = "", size_t cnt = 1) override;
            ISaveVisitor&
            operator()(const unsigned long long* val, const std::string& name = "", size_t cnt = 1) override;
#endif
#else
            ISaveVisitor& operator()(const long int* val, const std::string& name = "", size_t cnt = 1) override;
            ISaveVisitor&
            operator()(const unsigned long int* val, const std::string& name = "", size_t cnt = 1) override;
#endif
            ISaveVisitor& operator()(const float* val, const std::string& name = "", size_t cnt = 1) override;
            ISaveVisitor& operator()(const double* val, const std::string& name = "", size_t cnt = 1) override;
            ISaveVisitor& operator()(const void* binary, const std::string& name = "", size_t bytes = 1) override;

            ISaveVisitor& operator()(const IStructTraits* trait,
                                     const void* inst,
                                     const std::string& name = "",
                                     size_t cnt = 1) override;
            ISaveVisitor& operator()(const IContainerTraits* trait,
                                     const void* inst,
                                     const std::string& name = "",
                                     size_t cnt = 1) override;

            boost::python::object getObject();

            struct Shortcut
            {
                boost::python::object& obj;
                const std::string& name;
                bool used = false;
            };

          private:
            template <class T>
            void save(const T* val, const std::string& name, size_t cnt);
            boost::python::object m_object;
            std::unique_ptr<boost::python::list> m_list;
            // When copying out a single member, we use this to shortcut part of the serialization process
            // such that we don't have additional levels of stuff

            Shortcut* m_shortcut = nullptr;

            const python::DataConversionTable* m_conversion_table = nullptr;
        };

        struct ControlParamGetter : ToPythonVisitor
        {
        };

        struct FromPythonVisitor : LoadCache
        {
            FromPythonVisitor(const boost::python::object& obj,
                              const python::DataConversionTable* = python::DataConversionTable::instance());

            VisitorTraits traits() const override;

            std::shared_ptr<Allocator> getAllocator() const override;
            void setAllocator(std::shared_ptr<Allocator>) override;

            ILoadVisitor& operator()(bool* val, const std::string& name = "", size_t cnt = 1) override;
            ILoadVisitor& operator()(char* val, const std::string& name = "", size_t cnt = 1) override;
            ILoadVisitor& operator()(int8_t* val, const std::string& name = "", size_t cnt = 1) override;
            ILoadVisitor& operator()(uint8_t* val, const std::string& name = "", size_t cnt = 1) override;
            ILoadVisitor& operator()(int16_t* val, const std::string& name = "", size_t cnt = 1) override;
            ILoadVisitor& operator()(uint16_t* val, const std::string& name = "", size_t cnt = 1) override;
            ILoadVisitor& operator()(int32_t* val, const std::string& name = "", size_t cnt = 1) override;
            ILoadVisitor& operator()(uint32_t* val, const std::string& name = "", size_t cnt = 1) override;
            ILoadVisitor& operator()(int64_t* val, const std::string& name = "", size_t cnt = 1) override;
            ILoadVisitor& operator()(uint64_t* val, const std::string& name = "", size_t cnt = 1) override;
#ifdef ENVIRONMENT64
#ifndef _MSC_VER
            ILoadVisitor& operator()(long long* val, const std::string& name = "", size_t cnt = 1) override;
            ILoadVisitor& operator()(unsigned long long* val, const std::string& name = "", size_t cnt = 1) override;
#endif
#else
            ILoadVisitor& operator()(long int* val, const std::string& name = "", size_t cnt = 1) override;
            ILoadVisitor& operator()(unsigned long int* val, const std::string& name = "", size_t cnt = 1) override;
#endif
            ILoadVisitor& operator()(float* val, const std::string& name = "", size_t cnt = 1) override;
            ILoadVisitor& operator()(double* val, const std::string& name = "", size_t cnt = 1) override;
            ILoadVisitor& operator()(void* binary, const std::string& name = "", size_t bytes = 1) override;

            std::string getCurrentElementName() const override;
            size_t getCurrentContainerSize() const override;

            ILoadVisitor&
            operator()(const IStructTraits* trait, void* inst, const std::string& name = "", size_t cnt = 1) override;
            ILoadVisitor& operator()(const IContainerTraits* trait,
                                     void* inst,
                                     const std::string& name = "",
                                     size_t cnt = 1) override;

          protected:
            const boost::python::object& getObject() const;

            template <class T>
            void extract(T* val, const std::string& name = "", size_t cnt = 1);

          private:
            boost::python::object m_object;
            std::string m_current_object_name;
            const python::DataConversionTable* m_conversion_table = nullptr;
            bool m_loading_data = false;
        };

        struct ControlParamSetter : FromPythonVisitor
        {
            ControlParamSetter(const boost::python::object& obj);

            bool success() const
            {
                return true;
            }

            ILoadVisitor&
            operator()(const IStructTraits* trait, void* inst, const std::string& name = "", size_t cnt = 1) override;
        };

    } // namespace python
} // namespace mo
#endif // MO_PYTHON_PYTHON_CONVERSION_VISITATION_HPP