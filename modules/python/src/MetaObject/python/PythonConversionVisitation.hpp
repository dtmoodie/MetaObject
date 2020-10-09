#ifndef MO_PYTHON_PYTHON_CONVERSION_VISITATION_HPP
#define MO_PYTHON_PYTHON_CONVERSION_VISITATION_HPP
#include <MetaObject/runtime_reflection/DynamicVisitor.hpp>
#include <boost/python.hpp>

namespace boost
{
    namespace python
    {
        bool hasattr(const object& o, const char* name);
    }
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
            ToPythonVisitor();

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

            ISaveVisitor&
            operator()(IStructTraits* trait, const void* inst, const std::string& name = "", size_t cnt = 1) override;
            ISaveVisitor& operator()(IContainerTraits* trait,
                                     const void* inst,
                                     const std::string& name = "",
                                     size_t cnt = 1) override;

            boost::python::object getObject();

          private:
            template <class T>
            void save(const T* val, const std::string& name, size_t cnt);
            boost::python::object m_object;
            boost::python::list m_list;
            std::vector<boost::python::object> m_sub_object_stack;
            const python::DataConversionTable* m_conversion_table;
        };

        struct ControlParamGetter : ToPythonVisitor
        {
        };

        struct FromPythonVisitor : LoadCache
        {
            FromPythonVisitor(const boost::python::object& obj);

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
            operator()(IStructTraits* trait, void* inst, const std::string& name = "", size_t cnt = 1) override;
            ILoadVisitor&
            operator()(IContainerTraits* trait, void* inst, const std::string& name = "", size_t cnt = 1) override;

          protected:
            const boost::python::object& getObject() const;

            template <class T>
            void extract(T* val, const std::string& name = "", size_t cnt = 1);

          private:
            const boost::python::object& m_object;
            std::vector<boost::python::object> m_sub_object_stack;
            const python::DataConversionTable* m_conversion_table;
        };

        struct ControlParamSetter : FromPythonVisitor
        {
            ControlParamSetter(const boost::python::object& obj);

            bool success() const
            {
                return true;
            }

            ILoadVisitor&
            operator()(IStructTraits* trait, void* inst, const std::string& name = "", size_t cnt = 1) override;
        };

    } // namespace python
} // namespace mo
#endif // MO_PYTHON_PYTHON_CONVERSION_VISITATION_HPP