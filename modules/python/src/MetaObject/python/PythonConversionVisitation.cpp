#include "PythonConversionVisitation.hpp"
#include "DataConverter.hpp"
namespace mo
{
    namespace python
    {
        ToPythonVisitor::ToPythonVisitor()
            : m_conversion_table(python::DataConversionTable::instance())
        {
            MO_ASSERT(m_conversion_table != nullptr);
            m_object = boost::python::dict();
        }

        VisitorTraits ToPythonVisitor::traits() const
        {
            return {};
        }

        std::shared_ptr<Allocator> ToPythonVisitor::getAllocator() const
        {
            return {};
        }

        void ToPythonVisitor::setAllocator(std::shared_ptr<Allocator>)
        {
        }

        ISaveVisitor& ToPythonVisitor::operator()(const bool* val, const std::string& name, size_t cnt)
        {
            if (cnt == 1)
            {
                m_object[name] = *val;
            }
            else
            {
                for (size_t i = 0; i < cnt; ++i)
                {
                    m_list.append(val[i]);
                }
            }
        }

        ISaveVisitor& ToPythonVisitor::operator()(const char* val, const std::string& name, size_t cnt)
        {
            if (cnt == 1)
            {
                m_object[name] = *val;
            }
            else
            {
                for (size_t i = 0; i < cnt; ++i)
                {
                    m_list.append(val[i]);
                }
            }
        }

        ISaveVisitor& ToPythonVisitor::operator()(const int8_t* val, const std::string& name, size_t cnt)
        {
            if (cnt == 1)
            {
                m_object[name] = *val;
            }
            else
            {
                for (size_t i = 0; i < cnt; ++i)
                {
                    m_list.append(val[i]);
                }
            }
        }

        ISaveVisitor& ToPythonVisitor::operator()(const uint8_t* val, const std::string& name, size_t cnt)
        {
            if (cnt == 1)
            {
                m_object[name] = *val;
            }
            else
            {
                for (size_t i = 0; i < cnt; ++i)
                {
                    m_list.append(val[i]);
                }
            }
        }

        ISaveVisitor& ToPythonVisitor::operator()(const int16_t* val, const std::string& name, size_t cnt)
        {
            if (cnt == 1)
            {
                m_object[name] = *val;
            }
            else
            {
                for (size_t i = 0; i < cnt; ++i)
                {
                    m_list.append(val[i]);
                }
            }
        }

        ISaveVisitor& ToPythonVisitor::operator()(const uint16_t* val, const std::string& name, size_t cnt)
        {
            if (cnt == 1)
            {
                m_object[name] = *val;
            }
            else
            {
                for (size_t i = 0; i < cnt; ++i)
                {
                    m_list.append(val[i]);
                }
            }
        }

        ISaveVisitor& ToPythonVisitor::operator()(const int32_t* val, const std::string& name, size_t cnt)
        {
            if (cnt == 1)
            {
                m_object[name] = *val;
            }
            else
            {
                for (size_t i = 0; i < cnt; ++i)
                {
                    m_list.append(val[i]);
                }
            }
        }

        ISaveVisitor& ToPythonVisitor::operator()(const uint32_t* val, const std::string& name, size_t cnt)
        {
            if (cnt == 1)
            {
                m_object[name] = *val;
            }
            else
            {
                for (size_t i = 0; i < cnt; ++i)
                {
                    m_list.append(val[i]);
                }
            }
        }

        ISaveVisitor& ToPythonVisitor::operator()(const int64_t* val, const std::string& name, size_t cnt)
        {
            if (cnt == 1)
            {
                m_object[name] = *val;
            }
            else
            {
                for (size_t i = 0; i < cnt; ++i)
                {
                    m_list.append(val[i]);
                }
            }
        }

        ISaveVisitor& ToPythonVisitor::operator()(const uint64_t* val, const std::string& name, size_t cnt)
        {
            if (cnt == 1)
            {
                m_object[name] = *val;
            }
            else
            {
                for (size_t i = 0; i < cnt; ++i)
                {
                    m_list.append(val[i]);
                }
            }
        }

#ifdef ENVIRONMENT64
#ifndef _MSC_VER
        ISaveVisitor& ToPythonVisitor::operator()(const long long* val, const std::string& name, size_t cnt)
        {
            if (cnt == 1)
            {
                m_object[name] = *val;
            }
            else
            {
                for (size_t i = 0; i < cnt; ++i)
                {
                    m_list.append(val[i]);
                }
            }
        }

        ISaveVisitor& ToPythonVisitor::operator()(const unsigned long long* val, const std::string& name, size_t cnt)
        {
            if (cnt == 1)
            {
                m_object[name] = *val;
            }
            else
            {
                for (size_t i = 0; i < cnt; ++i)
                {
                    m_list.append(val[i]);
                }
            }
        }
#endif
#else
        ISaveVisitor& ToPythonVisitor::operator()(const long int* val, const std::string& name, size_t cnt)
        {
            if (cnt == 1)
            {
                m_object[name] = *val;
            }
            else
            {
                for (size_t i = 0; i < cnt; ++i)
                {
                    m_list.append(val[i]);
                }
            }
        }

        ISaveVisitor& ToPythonVisitor::operator()(const unsigned long int* val, const std::string& name, size_t cnt)
        {
            if (cnt == 1)
            {
                m_object[name] = *val;
            }
            else
            {
                for (size_t i = 0; i < cnt; ++i)
                {
                    m_list.append(val[i]);
                }
            }
        }
#endif

        ISaveVisitor& ToPythonVisitor::operator()(const float* val, const std::string& name, size_t cnt)
        {
            if (cnt == 1)
            {
                m_object[name] = *val;
            }
            else
            {
                for (size_t i = 0; i < cnt; ++i)
                {
                    m_list.append(val[i]);
                }
            }
        }

        ISaveVisitor& ToPythonVisitor::operator()(const double* val, const std::string& name, size_t cnt)
        {
            if (cnt == 1)
            {
                m_object[name] = *val;
            }
            else
            {
                for (size_t i = 0; i < cnt; ++i)
                {
                    m_list.append(val[i]);
                }
            }
        }

        ISaveVisitor& ToPythonVisitor::operator()(const void* binary, const std::string& name, size_t bytes)
        {
        }

        ISaveVisitor& ToPythonVisitor::
        operator()(IStructTraits* trait, const void* inst, const std::string& name, size_t cnt)
        {
            const mo::TypeInfo type = trait->type();
            python::DataConversionTable::ToPython_t converter = m_conversion_table->getConverterToPython(type);
            if (converter)
            {
                // If there is a specialized converter for this datatype, use it
                m_object[name] = converter(inst, trait);
            }
            else
            {
                m_sub_object_stack.push_back(std::move(m_object));
                m_object = boost::python::dict();
                trait->save(*this, inst, name, cnt);
                boost::python::object old_object = std::move(m_sub_object_stack.back());
                m_sub_object_stack.pop_back();
                old_object[name] = std::move(m_object);
                m_object = std::move(old_object);
            }
        }

        ISaveVisitor& ToPythonVisitor::
        operator()(IContainerTraits* trait, const void* inst, const std::string& name, size_t cnt)
        {
            const mo::TypeInfo type = trait->type();
            python::DataConversionTable::ToPython_t converter = m_conversion_table->getConverterToPython(type);
            if (converter)
            {
                m_object[name] = converter(inst, trait);
            }
            else
            {
                if (trait->valueType() == mo::TypeInfo::create<std::string>() &&
                    trait->keyType() == mo::TypeInfo::Void())
                {
                    const char* char_ptr = static_cast<const char*>(trait->valuePointer(inst));
                    const size_t size = trait->getContainerSize(inst);
                    m_object[name] = boost::python::str(char_ptr, size);
                }
                else
                {
                    m_sub_object_stack.push_back(std::move(m_object));
                    m_list = boost::python::list();
                    trait->save(*this, inst, name, cnt);
                    boost::python::object old_object = std::move(m_sub_object_stack.back());
                    m_sub_object_stack.pop_back();
                    old_object[name] = std::move(m_list);
                    m_object = std::move(old_object);
                }
            }
        }

        boost::python::object ToPythonVisitor::getObject()
        {
            return std::move(m_object);
        }

        // From python to C++

        FromPythonVisitor::FromPythonVisitor(const boost::python::object& obj)
            : m_object(obj)
        {
        }

        std::shared_ptr<Allocator> FromPythonVisitor::getAllocator() const
        {
        }

        void FromPythonVisitor::setAllocator(std::shared_ptr<Allocator>)
        {
        }

        VisitorTraits FromPythonVisitor::traits() const
        {
        }

        ILoadVisitor& FromPythonVisitor::operator()(bool* val, const std::string& name, size_t cnt)
        {
        }

        ILoadVisitor& FromPythonVisitor::operator()(char* val, const std::string& name, size_t cnt)
        {
        }

        ILoadVisitor& FromPythonVisitor::operator()(int8_t* val, const std::string& name, size_t cnt)
        {
        }

        ILoadVisitor& FromPythonVisitor::operator()(uint8_t* val, const std::string& name, size_t cnt)
        {
        }

        ILoadVisitor& FromPythonVisitor::operator()(int16_t* val, const std::string& name, size_t cnt)
        {
        }

        ILoadVisitor& FromPythonVisitor::operator()(uint16_t* val, const std::string& name, size_t cnt)
        {
        }

        ILoadVisitor& FromPythonVisitor::operator()(int32_t* val, const std::string& name, size_t cnt)
        {
        }

        ILoadVisitor& FromPythonVisitor::operator()(uint32_t* val, const std::string& name, size_t cnt)
        {
        }

        ILoadVisitor& FromPythonVisitor::operator()(int64_t* val, const std::string& name, size_t cnt)
        {
        }

        ILoadVisitor& FromPythonVisitor::operator()(uint64_t* val, const std::string& name, size_t cnt)
        {
        }
#ifdef ENVIRONMENT64
#ifndef _MSC_VER
        ILoadVisitor& FromPythonVisitor::operator()(long long* val, const std::string& name, size_t cnt)
        {
        }

        ILoadVisitor& FromPythonVisitor::operator()(unsigned long long* val, const std::string& name, size_t cnt)
        {
        }
#endif
#else
        ILoadVisitor& FromPythonVisitor::operator()(long int* val, const std::string& name, size_t cnt)
        {
        }

        ILoadVisitor& FromPythonVisitor::operator()(unsigned long int* val, const std::string& name, size_t cnt)
        {
        }
#endif
        ILoadVisitor& FromPythonVisitor::operator()(float* val, const std::string& name, size_t cnt)
        {
        }

        ILoadVisitor& FromPythonVisitor::operator()(double* val, const std::string& name, size_t cnt)
        {
        }

        ILoadVisitor& FromPythonVisitor::operator()(void* binary, const std::string& name, size_t bytes)
        {
        }

        ILoadVisitor& FromPythonVisitor::
        operator()(IStructTraits* trait, void* inst, const std::string& name, size_t cnt)
        {
        }

        ILoadVisitor& FromPythonVisitor::
        operator()(IContainerTraits* trait, void* inst, const std::string& name, size_t cnt)
        {
        }

        std::string FromPythonVisitor::getCurrentElementName() const
        {

            return {};
        }

        size_t FromPythonVisitor::getCurrentContainerSize() const
        {
            return 0;
        }

        ControlParamSetter::ControlParamSetter(const boost::python::object& obj)
            : FromPythonVisitor(obj)
        {
        }

    } // namespace python
} // namespace mo