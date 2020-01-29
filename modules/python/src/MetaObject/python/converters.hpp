#ifndef MO_PYTHON_CONVERTERS_HPP
#define MO_PYTHON_CONVERTERS_HPP

#include "MetaObject/core/TypeTable.hpp"
#include "MetaObject/logging/logging.hpp"
#include "MetaObject/python/FunctionSignatureBuilder.hpp"
#include "MetaObject/python/converters.hpp"
#include "MetaObject/types/small_vec.hpp"

#include "ct/VariadicTypedef.hpp"

#include <ct/type_traits.hpp>
#include <ct/types/TArrayView.hpp>

#include "PythonSetup.hpp"

#include <boost/python.hpp>
#include <boost/python/args.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include "reflected_converters.hpp"

#include <type_traits>
struct SystemTable;
/*
namespace mo
{
    namespace python
    {

        template <class T>
        void pythonizeData(const char* name);

        template <class T, ct::index_t PRIORITY = 10, class ENABLE = void>
        struct PythonConverter : public PythonConverter<T, PRIORITY - 1, void>
        {
            PythonConverter(SystemTable* table, const char* name)
                : PythonConverter<T, PRIORITY - 1, void>(table, name)
            {
            }
        };

        template <class T>
        struct PythonConverter<T, 0, ct::EnableIf<!std::is_arithmetic<T>::value>>
        {
            PythonConverter(SystemTable* table, const char* name)
            {
                python::registerSetupFunction(table, std::bind(&python::pythonizeData<T>, name));
            }

            static bool convertFromPython(const boost::python::object& obj, T& result)
            {
                boost::python::extract<T> extractor(obj);
                if (extractor.check())
                {
                    result = extractor();
                    return true;
                }
                return false;
            }

            static boost::python::object convertToPython(const T& result)
            {
                return boost::python::object(result);
            }

            static void registerToPython(const char* name = nullptr)
            {
                if (name)
                {
                    MO_LOG(debug, "Unable to register {} to python", name);
                }
                else
                {
                    auto name_ = mo::TypeTable::instance()->typeToName(mo::TypeInfo::create<T>());
                    MO_LOG(debug, "Unable to register {} to python", name_);
                }
                // what do...
                boost::python::class_<T> bpobj(name);
            }
        };

        template <class T>
        struct PythonConverter<T, 0, ct::EnableIf<std::is_arithmetic<T>::value>>
        {
            PythonConverter(SystemTable*, const char*)
            {
            }

            static bool convertFromPython(const boost::python::object& obj, T& result)
            {
                boost::python::extract<T> extractor(obj);
                if (extractor.check())
                {
                    result = extractor();
                    return true;
                }
                return false;
            }

            static boost::python::object convertToPython(const T& result)
            {
                return boost::python::object(result);
            }

            static void registerToPython(const char* = nullptr)
            {
            }
        };

        template <class T>
        boost::python::object convertToPython(const T& data)
        {
            return PythonConverter<T>::convertToPython(data);
        }

        template <class T>
        inline auto convertToPython(const T*& ptr) -> boost::python::object
        {
            if (ptr)
            {
                return convertToPython(*ptr);
            }
            return {};
        }

        template <class T>
        bool convertFromPython(const boost::python::object& obj, T& data)
        {
            return PythonConverter<T>::convertFromPython(obj, data);
        }

        template <class T>
        struct PythonConverter<std::vector<T>, 2, void>
        {
            PythonConverter(SystemTable* table, const char* name = nullptr)
            {
                python::registerSetupFunction(table, std::bind(&python::pythonizeData<std::vector<T>>, name));
            }

            static std::string reprVec(const std::vector<T>& data)
            {
                std::stringstream ss;
                ss << '[';
                for (size_t i = 0; i < data.size(); ++i)
                {
                    if (i != 0)
                    {
                        ss << ", ";
                    }
                    ss << data[i];
                }
                ss << ']';
                return ss.str();
            }

            static void registerToPython(const char* name_ = nullptr)
            {
                pythonizeData<T>(nullptr);
                std::string name;
                if (name_)
                {
                    name = name_;
                }
                else
                {
                    name = mo::TypeTable::instance()->typeToName(mo::TypeInfo(typeid(T)));
                    name += "List";
                }

                boost::python::class_<std::vector<T>> bpobj(name.c_str());
                bpobj.def(boost::python::vector_indexing_suite<std::vector<T>>());
                bpobj.def("__repr__", &reprVec);
            }

            static boost::python::list convertToPython(const std::vector<T>& data)
            {
                boost::python::list list;
                for (const auto& item : data)
                {
                    list.append(mo::python::convertToPython(item));
                }
                return list;
            }

            static bool convertFromPython(const boost::python::object& obj, std::vector<T>& result)
            {
                const ssize_t len = boost::python::len(obj);
                result.resize(len);
                for (ssize_t i = 0; i < len; ++i)
                {
                    mo::python::convertFromPython(boost::python::object(obj[i]), result[i]);
                }
                return true;
            }

        }; // PythonConvert<std::vector<T>, 2, void>

        template <class T>
        struct PythonConverter<T, 2, ct::EnableIfReflected<T>>
        {
            PythonConverter(SystemTable* table, const char* name)
            {
                python::registerSetupFunction(table, std::bind(&python::pythonizeData<T>, name));
            }

            static std::string repr(const T& obj)
            {
                std::stringstream ss;
                ct::printStruct(ss, obj);
                return std::move(ss).str();
            }

            static void registerToPython(const char* name)
            {
                boost::python::class_<T> bpobj(
                    name ? name : mo::TypeTable::instance()->typeToName(mo::TypeInfo(typeid(T))).c_str());
                reflected::addPropertyHelper<T>(bpobj, ct::Reflect<T>::end());
                reflected::addInit<T>(bpobj);
                bpobj.def("__repr__", &repr);
            }

            static bool convertFromPython(const boost::python::object&, T&)
            {
                // TODO
                return false;
            }

            static boost::python::object convertToPython(const T& result)
            {
                return boost::python::object(result);
            }

        }; // PythonConverter<T, 2, ct::EnableIfReflected<T>>

        template <class K, class V>
        struct PythonConverter<std::map<K, V>, 2, void>
        {

            PythonConverter(SystemTable* table, const char* name)
            {
                python::registerSetupFunction(table, std::bind(&python::pythonizeData<std::map<K, V>>, name));
            }

            static void registerToPython(const char* name_ = nullptr)
            {
                pythonizeData<K>(nullptr);
                pythonizeData<V>(nullptr);
                std::string name;
                if (name_)
                {
                    name = name_;
                }
                else
                {
                    name = mo::TypeTable::instance()->typeToName(mo::TypeInfo(typeid(std::map<K, V>)));
                }
                boost::python::class_<std::map<K, V>> bpobj(name.c_str());
                bpobj.def(boost::python::map_indexing_suite<std::map<K, V>>());
            }

            static bool convertFromPython(const boost::python::object& obj, std::map<K, V>& result)
            {
                boost::python::extract<std::map<K, V>> extractor(obj);
                result = extractor();
                return true;
            }

            static boost::python::object convertToPython(const std::map<K, V>&)
            {
                boost::python::dict dict;
                // TODO

                return dict;
            }
        };

        template <class T, int N>
        struct PythonConverter<mo::SmallVec<T, N>, 2, void>
        {
            PythonConverter(SystemTable* table, const char* name = nullptr)
            {
                python::registerSetupFunction(table, std::bind(&python::pythonizeData<mo::SmallVec<T, N>>, name));
            }

            static boost::python::object convertToPython(const mo::SmallVec<T, N>& vec)
            {
                boost::python::list list;
                for (const auto& item : vec)
                {
                    list.append(convertToPython(item));
                }
                return list;
            }
        };

        template <class T>
        struct PythonConverter<ct::TArrayView<T>, 2, void>
        {
            PythonConverter(SystemTable* table, const char* name = nullptr)
            {
                python::registerSetupFunction(table, std::bind(&python::pythonizeData<ct::TArrayView<T>>, name));
            }

            static bool convertFromPython(const boost::python::object& obj, ct::TArrayView<T>& result)
            {
                const ssize_t len = boost::python::len(obj);
                if (len != result.size())
                {
                    return false;
                }
                for (ssize_t i = 0; i < len; ++i)
                {
                    if (!python::convertFromPython(obj[i], result[i]))
                    {
                        return false;
                    }
                }
                return true;
            }
            static boost::python::object convertToPython(const ct::TArrayView<T>& data)
            {
                boost::python::list list;
                for (const auto& item : data)
                {
                    list.append(python::convertToPython(item));
                }
                return list;
            }
        };

        template <>
        struct PythonConverter<ct::TArrayView<void>, 2, void>
        {
            PythonConverter(SystemTable*, const char* = nullptr)
            {
                // python::registerSetupFunction(table, std::bind(&python::pythonizeData<ct::TArrayView<void>>, name));
            }

            static bool convertFromPython(const boost::python::object& obj, ct::TArrayView<void>& result)
            {
                PyObject* py_buffer = obj.ptr();
                if (!PyBuffer_Check(py_buffer))
                {
                    return false;
                }

                void* cxx_buf = py_buffer->buf;
                const auto size = py_buffer->len;
                result = ct::TArrayView<void>(cxx_buf, size);
                return true;
            }
            static boost::python::object convertToPython(const ct::TArrayView<void>& data)
            {
                PyObject* py_buf = PyBuffer_FromMemory(const_cast<void*>(data.data()), data.size());
                return boost::python::object(boost::python::handle<>(py_buf));
            }
            static void registerToPython(const char* name)
            {
            }
        };

        template <>
        struct PythonConverter<ct::TArrayView<const void>, 2, void>
        {
            PythonConverter(SystemTable* table, const char* name = nullptr)
            {
                // python::registerSetupFunction(table, std::bind(&python::pythonizeData<ct::TArrayView<void>>, name));
            }

            static bool convertFromPython(const boost::python::object& obj, ct::TArrayView<const void>& result)
            {
                PyObject* py_buffer = obj.ptr();
                if (!PyBuffer_Check(py_buffer))
                {
                    return false;
                }

                void* cxx_buf = py_buffer->buf;
                const auto size = py_buffer->len;
                result = ct::TArrayView<void>(cxx_buf, size);
                return true;
            }
            static boost::python::object convertToPython(const ct::TArrayView<const void>& data)
            {
                PyObject* py_buf = PyBuffer_FromMemory(const_cast<void*>(data.data()), data.size());
                return boost::python::object(boost::python::handle<>(py_buf));
            }

            static void registerToPython(const char* name)
            {
            }
        };

        template <>
        struct PythonConverter<std::string, 3, void>
        {
            inline PythonConverter(SystemTable* table, const char* name)
            {
                python::registerSetupFunction(table, std::bind(&python::pythonizeData<std::string>, name));
            }

            static inline bool convertFromPython(const boost::python::object& obj, std::string& result)
            {
                boost::python::extract<std::string> extractor(obj);
                if (extractor.check())
                {
                    result = extractor();
                    return true;
                }
                return false;
            }

            static inline boost::python::object convertToPython(const std::string& result)
            {
                return boost::python::object(result);
            }

            static inline void registerToPython(const char*)
            {
                // Not sure if needed for an std::string
            }
        };

        template <class T>
        void pythonizeData(const char* name)
        {
            static bool registered = false;
            if (registered)
            {
                return;
            }
            auto module_name = getModuleName();
            boost::python::object datatype_module(boost::python::handle<>(
                boost::python::borrowed(PyImport_AddModule((module_name + ".datatypes").c_str()))));

            // set the current scope to the new sub-module
            boost::python::scope plugins_scope = datatype_module;

            PythonConverter<T>::registerToPython(name);
            registered = true;
        }
    }
}*/
#endif // MO_PYTHON_CONVERTERS_HPP
