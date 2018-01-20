#pragma once
#include "MetaObject/core/Demangle.hpp"
#include "MetaObject/core/detail/Counter.hpp"
#include "MetaObject/python/converters.hpp"
#include "MetaObject/python/MetaObject.hpp"
#include <MetaObject/logging/logging.hpp>

#include "ct/reflect/reflect_data.hpp"
#include "ct/reflect/printer.hpp"

#include "PythonSetup.hpp"

#include "boost/python.hpp"
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
                bpobj.add_property(ct::reflect::getName<0, T>(),&ct::reflect::getValue<0, T>, &ct::reflect::setValue<0, T>);
            }

            template <class T, int I, class BP>
            void addPropertyHelper(BP& bpobj, mo::_counter_<I>)
            {
                bpobj.add_property(ct::reflect::getName<I, T>(), &ct::reflect::getValue<I, T>, &ct::reflect::setValue<I, T>);
                addPropertyHelper<T>(bpobj, mo::_counter_<I - 1>());
            }
            template<class T, class V>
            V inferSetterType(void(*)(T&, V))
            {

            }

            template<int I, class T>
            using SetValue_t = decltype(inferSetterType(ct::reflect::setValue<I, T>));

            template<class T>
            void initDataMembers(T& data, boost::python::object& value)
            {
                typedef std::decay_t<SetValue_t<ct::reflect::ReflectData<T>::N - 1, T>> Type;
                boost::python::extract<Type> ext(value);
                if (ext.check())
                {
                    Type value = ext();
                    ct::reflect::setValue<ct::reflect::ReflectData<T>::N - 1>(data, value);
                }
            }

            template<class T, class... Args>
            void initDataMembers(T& data, boost::python::object& value, Args... args)
            {
                typedef std::decay_t<SetValue_t<ct::reflect::ReflectData<T>::N - sizeof...(args) - 1, T>> Type;
                boost::python::extract<Type> ext(value);
                if (ext.check())
                {
                    Type value = ext();
                    ct::reflect::setValue<ct::reflect::ReflectData<T>::N - sizeof...(args) - 1>(data, value);
                }
                initDataMembers(data, args...);
            }

            template<class ... Args>
            struct CreateDataObject{};

            template<class T, class ... Args>
            struct CreateDataObject<T, ce::variadic_typedef<Args...>>
            {
                static T* create(Args... args)
                {
                    T* obj = new T();
                    initDataMembers(*obj, args...);
                    return obj;
                }
            };

            template<class T, class BP>
            void addInit(BP& bpobj)
            {
                bpobj.def("__init__", boost::python::make_constructor(CreateDataObject<T, FunctionSignatureBuilder<boost::python::object, ct::reflect::ReflectData<T>::N - 1>::VariadicTypedef_t>::create, boost::python::default_call_policies()));
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
                static bool registered = false;
                if (registered)
                    return;
                boost::python::class_<T> bpobj(name ? name : mo::Demangle::typeToName(mo::TypeInfo(typeid(T))).c_str());
                addPropertyHelper<T>(bpobj, mo::_counter_<ct::reflect::ReflectData<T>::N - 1>());
                addInit<T>(bpobj);
                bpobj.def("__repr__", &repr<T>);
                registered = true;
            }

            template <class T>
            ct::reflect::enable_if_reflected<T> pythonizeDataHelper(const char* name_ = nullptr,
                                                                    const std::vector<T>* = nullptr)
            {
                static bool registered = false;
                if (registered)
                    return;
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
                registered = true;
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
