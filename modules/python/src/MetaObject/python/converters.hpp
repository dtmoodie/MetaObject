#pragma once
#include "MetaObject/core/Demangle.hpp"
#include "MetaObject/core/detail/Counter.hpp"
#include "MetaObject/logging/logging.hpp"
#include "MetaObject/python/FunctionSignatureBuilder.hpp"
#include "MetaObject/python/converters.hpp"

#include "ct/VariadicTypedef.hpp"
#include "ct/reflect/printer.hpp"
#include "ct/reflect/reflect_data.hpp"
#include <ct/detail/TypeTraits.hpp>

#include "PythonSetup.hpp"

#include <boost/python.hpp>
#include <boost/python/args.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <type_traits>

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
        struct PythonConverter;

        template <class T, class Enable = void>
        struct ToPythonDataConverter
        {
        };

        template <class T>
        struct ToPythonDataConverter<T, ct::reflect::enable_if_reflected<T>>
        {
            ToPythonDataConverter(SystemTable* /*table*/, const char* name)
            {
                python::registerSetupFunction(std::bind(&python::detail::pythonizeData<T>, name));
            }
        };

        template <class T>
        struct ToPythonDataConverter<T,
                                     typename std::enable_if<!ct::reflect::ReflectData<T, void>::IS_SPECIALIZED &&
                                                             !ct::reflect::is_container<T>::value>::type>
        {
            ToPythonDataConverter(SystemTable* /*table*/, const char* name)
            {
                python::registerSetupFunction(std::bind(&python::detail::pythonizeData<T>, name));
            }
        };

        template <class K, class V>
        struct ToPythonDataConverter<std::map<K, V>, void>
        {
            ToPythonDataConverter(SystemTable* /*table*/, const char* name)
            {
                python::registerSetupFunction(std::bind(&python::detail::pythonizeData<std::map<K, V>>, name));
            }
        };

        template <class T>
        struct ToPythonDataConverter<std::vector<T>, ct::reflect::enable_if_reflected<T>>
        {
            ToPythonDataConverter(SystemTable* /*table*/, const char* name)
            {
                python::registerSetupFunction(std::bind(&python::detail::pythonizeData<std::vector<T>>, name));
            }
        };

        template <>
        struct ToPythonDataConverter<std::string, void>
        {
            ToPythonDataConverter(SystemTable* /*table*/, const char* name)
            {
                python::registerSetupFunction(std::bind(&python::detail::pythonizeData<std::string>, name));
            }
        };

        template <class T>
        struct ToPythonDataConverter<std::vector<T>, ct::reflect::enable_if_not_reflected<T>>
        {
            ToPythonDataConverter(SystemTable* /*table*/, const char* name)
            {
                python::registerSetupFunction(std::bind(&python::detail::pythonizeData<std::vector<T>>, name));
            }
        };

        template <class T>
        inline auto convertToPython(const T& obj) -> boost::python::object
        {
            return boost::python::object(obj);
        }

        template <class T>
        inline auto convertToPython(const std::vector<T>& vec) -> boost::python::object
        {
            boost::python::list list;
            for (const auto& item : vec)
            {
                list.append(convertToPython(item));
            }
            return list;
        }

        inline void convertFromPython(const boost::python::object& obj, std::string& result);

        template <class K, class T>
        inline void convertFromPython(const boost::python::object& obj, std::map<K, T>& result);

        template <class T>
        inline auto convertFromPython(const boost::python::object& obj, T& result)
            -> ct::reflect::enable_if_not_reflected<T>;

        template <class T>
        inline auto convertFromPython(const boost::python::object& obj, T& result)
            -> ct::reflect::enable_if_reflected<T>;

        template <class T>
        inline void convertFromPython(const boost::python::object& obj, std::vector<T>& result);

        namespace detail
        {
            template <class T, class V>
            V inferSetterType(void (*)(T&, V))
            {
            }

            template <class T, class V>
            V inferGetterType(V (*)(T&))
            {
            }

            template <int I, class T>
            using SetValue_t = decltype(inferSetterType<T>(ct::reflect::setValue<I, T>));

            template <int I, class T>
            using GetValue_t = decltype(inferGetterType<T>(ct::reflect::getValue<I, T>));

            template <class T, int I, class BP>
            auto addPropertyImpl(BP& bpobj) -> typename std::enable_if<
                !std::is_pointer<decltype(ct::reflect::getValue<I>(std::declval<T>()))>::value>::type
            {
                bpobj.add_property(
                    ct::reflect::getName<I, T>(), &ct::reflect::getValue<I, T>, &ct::reflect::setValue<I, T>);
            }

            template <class T, int I, class BP>
            auto addPropertyImpl(BP& bpobj) -> typename std::enable_if<
                std::is_pointer<decltype(ct::reflect::getValue<I>(std::declval<T>()))>::value>::type
            {
                // bpobj.add_property(
                //    ct::reflect::getName<I, T>(), &ct::reflect::getValue<I, T>, &ct::reflect::setValue<I, T>);
            }

            template <class T, class BP>
            void addPropertyHelper(BP& bpobj, mo::_counter_<0>)
            {
                addPropertyImpl<T, 0>(bpobj);
            }

            template <class T, int I, class BP>
            void addPropertyHelper(BP& bpobj, mo::_counter_<I>)
            {
                addPropertyImpl<T, I>(bpobj);
                addPropertyHelper<T>(bpobj, mo::_counter_<I - 1>());
            }

            template <class T>
            void initDataMembers(T& data, boost::python::object& value)
            {
                typedef typename std::decay<SetValue_t<ct::reflect::ReflectData<T>::N - 1, T>>::type Type;
                boost::python::extract<Type> ext(value);
                if (ext.check())
                {
                    Type value = ext();
                    ct::reflect::setValue<ct::reflect::ReflectData<T>::N - 1>(data, std::move(value));
                }
            }

            template <class T, class... Args>
            void initDataMembers(T& data, boost::python::object& value, Args... args)
            {
                typedef
                    typename std::decay<SetValue_t<ct::reflect::ReflectData<T>::N - sizeof...(args)-1, T>>::type Type;
                boost::python::extract<Type> ext(value);
                if (ext.check())
                {
                    Type value = ext();
                    ct::reflect::setValue<ct::reflect::ReflectData<T>::N - sizeof...(args)-1>(data, std::move(value));
                }
                initDataMembers(data, args...);
            }

            template <class... Args>
            struct CreateDataObject
            {
            };

            template <class T>
            void populateKwargs(std::array<boost::python::detail::keyword, ct::reflect::ReflectData<T>::N>& kwargs,
                                ct::_counter_<0> cnt)
            {
                kwargs[0] = (boost::python::arg(ct::reflect::getName<0, T>()) = boost::python::object());
            }

            template <class T, int I>
            void populateKwargs(std::array<boost::python::detail::keyword, ct::reflect::ReflectData<T>::N>& kwargs,
                                ct::_counter_<I> cnt,
                                typename std::enable_if<I != 0>::type* = 0)
            {
                kwargs[I] = (boost::python::arg(ct::reflect::getName<I, T>()) = boost::python::object());
                populateKwargs<T>(kwargs, ct::_counter_<I - 1>());
            }

            template <class T>
            std::array<boost::python::detail::keyword, ct::reflect::ReflectData<T>::N> makeKeywordArgs()
            {
                std::array<boost::python::detail::keyword, ct::reflect::ReflectData<T>::N> kwargs;
                populateKwargs<T>(kwargs, ct::_counter_<ct::reflect::ReflectData<T>::N - 1>());
                return kwargs;
            }

            template <class T, class... Args>
            struct CreateDataObject<T, ct::variadic_typedef<Args...>>
            {
                static const int size = ct::reflect::ReflectData<T>::N - 1;
                static T* create(Args... args)
                {
                    T* obj = new T();
                    initDataMembers(*obj, args...);
                    return obj;
                }

                static boost::python::detail::keyword_range range()
                {
                    static std::array<boost::python::detail::keyword, ct::reflect::ReflectData<T>::N> s_keywords =
                        makeKeywordArgs<T>();
                    return std::make_pair<boost::python::detail::keyword const*, boost::python::detail::keyword const*>(
                        &s_keywords[0], &s_keywords[0] + ct::reflect::ReflectData<T>::N);
                }
            };

            template <class T, class BP>
            typename std::enable_if<ct::is_default_constructible<T>::value>::type addInit(BP& bpobj)
            {
                typedef CreateDataObject<
                    T,
                    typename FunctionSignatureBuilder<boost::python::object,
                                                      ct::reflect::ReflectData<T>::N - 1>::VariadicTypedef_t>
                    Creator_t;
                bpobj.def("__init__",
                          boost::python::make_constructor(
                              Creator_t::create, boost::python::default_call_policies(), Creator_t()));
            }

            template <class T, class BP>
            typename std::enable_if<!ct::is_default_constructible<T>::value>::type addInit(BP&)
            {
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
                    convertFromPython(python_member, ct::reflect::get<0>(obj));
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
                    convertFromPython(python_member, ct::reflect::get<I>(obj));
                }
                extract(obj, bpobj, mo::_counter_<I - 1>());
            }

            template <class T>
            ct::reflect::enable_if_not_reflected<T> pythonizeDataHelper(const char* /*name*/ = nullptr,
                                                                        const T* = nullptr)
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
                boost::python::class_<T> bpobj(name ? name : mo::Demangle::typeToName(mo::TypeInfo(typeid(T))).c_str(),
                                               boost::python::no_init);
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
                auto module_name = getModuleName();
                boost::python::object datatype_module(boost::python::handle<>(
                    boost::python::borrowed(PyImport_AddModule((module_name + ".datatypes").c_str()))));
                // boost::python::scope().attr("datatypes") = datatype_module;
                // set the current scope to the new sub-module
                boost::python::scope plugins_scope = datatype_module;

                pythonizeDataHelper(name, static_cast<const T*>(nullptr));
            }
        } // namespace mo::python::detail

        template <class K, class T>
        inline void convertFromPython(const boost::python::object& obj, std::map<K, T>& result)
        {
            boost::python::extract<std::map<K, T>> extractor(obj);
            result = extractor();
        }

        template <class T>
        inline auto convertFromPython(const boost::python::object& obj, T& result)
            -> ct::reflect::enable_if_not_reflected<T>
        {
            boost::python::extract<T> extractor(obj);
            result = extractor();
        }

        template <class T>
        inline auto convertFromPython(const boost::python::object& obj, T& result)
            -> ct::reflect::enable_if_reflected<T>
        {
            detail::extract(result, obj, mo::_counter_<ct::reflect::ReflectData<T>::N - 1>());
        }

        inline void convertFromPython(const boost::python::object& obj, std::string& result)
        {
            result = boost::python::extract<std::string>(obj)();
        }

        template <class T>
        inline void convertFromPython(const boost::python::object& obj, ct::reflect::enable_if_reflected<T>& result)
        {
            detail::extract(result, obj, mo::_counter_<ct::reflect::ReflectData<T>::N - 1>());
        }

        template <class T>
        inline void convertFromPython(const boost::python::object& obj, std::vector<T>& result)
        {
            const ssize_t len = boost::python::len(obj);
            result.resize(len);
            for (ssize_t i = 0; i < len; ++i)
            {
                convertFromPython(boost::python::object(obj[i]), result[i]);
            }
        }
    }
}

namespace boost
{
    namespace python
    {
        namespace detail
        {
            template <class... T>
            struct is_keywords<mo::python::detail::CreateDataObject<T...>>
            {
                static const bool value = true;
            };
        }
    }
}

#include "detail/converters.hpp"
