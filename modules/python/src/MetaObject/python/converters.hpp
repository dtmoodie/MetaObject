#pragma once
#include "MetaObject/core/Demangle.hpp"
#include "MetaObject/core/detail/Counter.hpp"
#include "ct/reflect/reflect_data.hpp"
#include "ct/reflect/printer.hpp"
#include "MetaObject/python/converters.hpp"
#include "PythonSetup.hpp"
#include "boost/python.hpp"
#include <MetaObject/logging/logging.hpp>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
namespace mo
{
    namespace python
    {
        namespace detail
        {
            template <class T>
            void pythonizeData(const char* name);
        }

        template <class T, class Enable = void>
        struct ToPythonDataConverter
        {
        };

        template <class T>
        struct ToPythonDataConverter<T, ct::reflect::enable_if_reflected<T>>
        {
            ToPythonDataConverter()
            {
                python::registerSetupFunction(std::bind(&python::detail::pythonizeData<T>, nullptr));
            }
        };

        template <class T>
        struct ToPythonDataConverter<std::vector<T>, ct::reflect::enable_if_reflected<T>>
        {
            ToPythonDataConverter()
            {
                python::registerSetupFunction(std::bind(&python::detail::pythonizeData<std::vector<T>>, nullptr));
            }
        };

        inline void convertFromPython(std::string& result, const boost::python::object& obj);

        template <class K, class T>
        inline void convertFromPython(std::map<K, T>& result, const boost::python::object& obj);

        template <class T>
        inline ct::reflect::enable_if_not_reflected<T> convertFromPython(T& result, const boost::python::object& obj);

        template <class T>
        inline ct::reflect::enable_if_reflected<T> convertFromPython(T& result, const boost::python::object& obj);

        template <class T>
        inline void convertFromPython(std::vector<T>& result, const boost::python::object& obj);

        namespace detail
        {
            template <class T, class BP>
            void addPropertyHelper(BP& bpobj, mo::_counter_<0>)
            {
                using Return_t = std::decay_t<decltype(ct::reflect::get<0, T>(std::declval<T const>()))>;
                bpobj.add_property(ct::reflect::getName<0, T>(),
                                   static_cast<Return_t (*)(const T&)>(&ct::reflect::getValue<0, T>));
            }

            template <class T, int I, class BP>
            void addPropertyHelper(BP& bpobj, mo::_counter_<I>)
            {
                using Return_t = std::decay_t<decltype(ct::reflect::get<I, T>(std::declval<T const>()))>;
                bpobj.add_property(ct::reflect::getName<I, T>(),
                                   static_cast<Return_t (*)(const T&)>(&ct::reflect::getValue<I, T>));
                addPropertyHelper<T>(bpobj, mo::_counter_<I - 1>());
            }

            template <class T>
            void extract(T& obj, const boost::python::object& bpobj, mo::_counter_<0>)
            {
                auto ptr = bpobj.ptr();
                boost::python::object python_member;
                if (PyObject_HasAttrString(ptr, ct::reflect::getName<0, T>()))
                {
                    python_member = bpobj.attr(ct::reflect::getName<0, T>());
                }
                else
                {
                    if (0 < boost::python::len(bpobj))
                    {
                        python_member = bpobj[0];
                    }
                }
                if (python_member)
                {
                    convertFromPython(ct::reflect::get<0>(obj), python_member);
                }
            }

            template <class T, int I>
            void extract(T& obj, const boost::python::object& bpobj, mo::_counter_<I>)
            {
                auto ptr = bpobj.ptr();
                boost::python::object python_member;
                if (PyObject_HasAttrString(ptr, ct::reflect::getName<I, T>()))
                {
                    python_member = bpobj.attr(ct::reflect::getName<I, T>());
                }
                else
                {
                    if (I < boost::python::len(bpobj))
                    {
                        python_member = bpobj[I];
                    }
                }
                if (python_member)
                {
                    convertFromPython(ct::reflect::get<I>(obj), python_member);
                }
                extract(obj, bpobj, mo::_counter_<I - 1>());
            }

            template <class T>
            ct::reflect::enable_if_not_reflected<T> pythonizeDataHelper(const char* name = nullptr, const T* = nullptr)
            {
            }

            template <class T>
            ct::reflect::enable_if_reflected<T, std::string> repr(const T& data)
            {
                std::stringstream ss;
                ct::reflect::printStruct(ss, data);
                return ss.str();
            }

            template <class T>
            ct::reflect::enable_if_reflected<T, std::string> reprVec(const std::vector<T>& data)
            {
                std::stringstream ss;
                ss << '[';
                for (size_t i = 0; i < data.size(); ++i)
                {
                    if (i != 0)
                        ss << ", ";
                    ct::reflect::printStruct(ss, data[i]);
                }
                ss << ']';
                return ss.str();
            }

            template <class T>
            ct::reflect::enable_if_reflected<T> pythonizeDataHelper(const char* name = nullptr, const T* = nullptr)
            {
                boost::python::class_<T> bpobj(name ? name : mo::Demangle::typeToName(mo::TypeInfo(typeid(T))).c_str());
                detail::addPropertyHelper<T>(bpobj, mo::_counter_<ct::reflect::ReflectData<T>::N - 1>());
                bpobj.def("__repr__", &repr<T>);
            }

            template <class T>
            ct::reflect::enable_if_reflected<T> pythonizeDataHelper(const char* name_ = nullptr,
                                                                    const std::vector<T>* = nullptr)
            {
                pythonizeDataHelper<T>(name_, static_cast<const T*>(nullptr));
                std::string name;
                if (name_)
                    name = name_;
                else
                    name = mo::Demangle::typeToName(mo::TypeInfo(typeid(T)));
                name += "List";
                boost::python::class_<std::vector<T>> bpobj(name.c_str());
                bpobj.def(boost::python::vector_indexing_suite<std::vector<T>>());
                bpobj.def("__repr__", &reprVec<T>);
            }

            template <class K, class T>
            ct::reflect::enable_if_reflected<T> pythonizeDataHelper(const char* name = nullptr,
                                                                    const std::map<K, T>* = nullptr)
            {
                pythonizeDataHelper<T>(name, static_cast<const T*>(nullptr));
                boost::python::class_<std::vector<T>> bpobj((std::string(name) + "Map").c_str());
                bpobj.def(boost::python::map_indexing_suite<std::map<K, T>>());
            }

            template <class T>
            void pythonizeData(const char* name)
            {
                boost::python::object plugins_module(
                    boost::python::handle<>(boost::python::borrowed(PyImport_AddModule("metaobject.datatypes"))));
                boost::python::scope().attr("datatypes") = plugins_module;
                // set the current scope to the new sub-module
                boost::python::scope plugins_scope = plugins_module;

                pythonizeDataHelper(name, static_cast<const T*>(nullptr));
            }
        } // namespace mo::python::detail

        inline void convertFromPython(std::string& result, const boost::python::object& obj)
        {
            boost::python::extract<std::string> extractor(obj);
            result = extractor();
        }

        template <class K, class T>
        inline void convertFromPython(std::map<K, T>& result, const boost::python::object& obj)
        {
            boost::python::extract<std::map<K, T>> extractor(obj);
            result = extractor();
        }

        template <class T>
        inline ct::reflect::enable_if_not_reflected<T> convertFromPython(T& result, const boost::python::object& obj)
        {
            boost::python::extract<T> extractor(obj);
            result = extractor();
        }

        template <class T>
        inline ct::reflect::enable_if_reflected<T> convertFromPython(T& result, const boost::python::object& obj)
        {
            detail::extract(result, obj, mo::_counter_<ct::reflect::ReflectData<T>::N - 1>());
        }

        template <class T>
        inline void convertFromPython(std::vector<T>& result, const boost::python::object& obj)
        {
            const ssize_t len = boost::python::len(obj);
            result.resize(len);
            for (ssize_t i = 0; i < len; ++i)
            {
                convertFromPython(result[i], boost::python::object(obj[i]));
            }
        }
    }
}

#include "detail/converters.hpp"
