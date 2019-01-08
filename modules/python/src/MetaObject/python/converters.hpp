#pragma once
#include "MetaObject/core/TypeTable.hpp"
#include "MetaObject/core/detail/Counter.hpp"
#include "MetaObject/logging/logging.hpp"
#include "MetaObject/python/FunctionSignatureBuilder.hpp"
#include "MetaObject/python/converters.hpp"
#include "MetaObject/types/small_vec.hpp"

#include "ct/VariadicTypedef.hpp"
#include "ct/reflect.hpp"
#include "ct/reflect/print.hpp"
#include <ct/TypeTraits.hpp>

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
        struct ToPythonDataConverter<T, ct::enable_if_reflected<T>>
        {
            ToPythonDataConverter(SystemTable* /*table*/, const char* name)
            {
                python::registerSetupFunction(std::bind(&python::detail::pythonizeData<T>, name));
            }
        };

        template <class T>
        struct ToPythonDataConverter<
            T,
            typename std::enable_if<!ct::Reflect<T>::SPECIALIZED && !ct::is_container<T>::value>::type>
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
        struct ToPythonDataConverter<std::vector<T>, ct::enable_if_reflected<T>>
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
        struct ToPythonDataConverter<std::vector<T>, ct::enable_if_not_reflected<T>>
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

        template <class T, int N>
        inline auto convertToPython(const mo::SmallVec<T, N>& vec) -> boost::python::object
        {
            boost::python::list list;
            for (const auto& item : vec)
            {
                list.append(convertToPython(item));
            }
            return list;
        }

        template <class T>
        inline auto convertToPython(const T*& ptr) -> boost::python::object
        {
            if (ptr)
            {
                return convertToPython(*ptr);
            }
            else
            {
                return {};
            }
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
        inline auto convertFromPython(const boost::python::object& obj, T& result) -> ct::enable_if_not_reflected<T>;

        template <class T>
        inline auto convertFromPython(const boost::python::object& obj, T& result) -> ct::enable_if_reflected<T>;

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
            using SetValue_t = typename ct::SetterType<T, I>::type;

            template <int I, class T>
            using GetValue_t = typename ct::GetterType<T, I>::type;

            template <class T, int I>
            boost::python::object pythonGet(const T& obj)
            {
                auto accessor = ct::Reflect<T>::getAccessor(ct::Indexer<I>{});
                return convertToPython(accessor.get(obj));
            }

            template <class T, int I>
            void pythonSet(T& obj, const SetValue_t<I, T>& val)
            {
                auto accessor = ct::Reflect<T>::getAccessor(ct::Indexer<I>{});
                accessor.set(obj, val);
            }

            template <class T, int I, class BP>
            auto addPropertyImpl(BP& bpobj) -> typename std::enable_if<!std::is_pointer<GetValue_t<I, T>>::value>::type
            {
                bpobj.add_property(ct::Reflect<T>::getName(ct::Indexer<I>{}), &pythonGet<T, I>, &pythonSet<T, I>);
            }

            template <class T, int I, class BP>
            auto addPropertyImpl(BP& bpobj) -> typename std::enable_if<std::is_pointer<GetValue_t<I, T>>::value>::type
            {
                bpobj.add_property(ct::Reflect<T>::getName(ct::Indexer<I>{}), &pythonGet<T, I>, &pythonSet<T, I>);
            }

            template <class T, class BP>
            void addPropertyHelper(BP& bpobj, const ct::Indexer<0>)
            {
                addPropertyImpl<T, 0>(bpobj);
            }

            template <class T, ct::index_t I, class BP>
            void addPropertyHelper(BP& bpobj, const ct::Indexer<I> idx)
            {
                addPropertyImpl<T, I>(bpobj);
                addPropertyHelper<T>(bpobj, --idx);
            }

            template <class T>
            void initDataMembers(T& data, boost::python::object& value)
            {

                typedef typename std::decay<SetValue_t<ct::Reflect<T>::N, T>>::type Type;
                boost::python::extract<Type> ext(value);
                if (ext.check())
                {
                    Type value = ext();
                    auto accessor = ct::Reflect<T>::getAccessor(ct::Indexer<ct::Reflect<T>::N>{});
                    accessor.set(data, std::move(value));
                }
            }

            template <class T, class... Args>
            void initDataMembers(T& data, boost::python::object& value, Args... args)
            {
                typedef typename std::decay<SetValue_t<ct::Reflect<T>::N - sizeof...(args), T>>::type Type;
                boost::python::extract<Type> ext(value);
                if (ext.check())
                {
                    Type value = ext();
                    auto accessor = ct::Reflect<T>::getAccessor(ct::Indexer<ct::Reflect<T>::N - sizeof...(args)>{});
                    accessor.set(data, std::move(value));
                }
                initDataMembers(data, args...);
            }

            template <class... Args>
            struct CreateDataObject
            {
            };

            template <class T>
            void populateKwargs(std::array<boost::python::detail::keyword, ct::Reflect<T>::N + 1>& kwargs,
                                const ct::Indexer<0> cnt)
            {
                kwargs[0] = (boost::python::arg(ct::Reflect<T>::getName(cnt)) = boost::python::object());
            }

            template <class T, ct::index_t I>
            void populateKwargs(std::array<boost::python::detail::keyword, ct::Reflect<T>::N + 1>& kwargs,
                                const ct::Indexer<I> cnt,
                                typename std::enable_if<I != 0>::type* = 0)
            {
                kwargs[I] = (boost::python::arg(ct::Reflect<T>::getName(cnt)) = boost::python::object());
                populateKwargs<T>(kwargs, --cnt);
            }

            template <class T>
            std::array<boost::python::detail::keyword, ct::Reflect<T>::N + 1> makeKeywordArgs()
            {
                std::array<boost::python::detail::keyword, ct::Reflect<T>::N + 1> kwargs;
                populateKwargs<T>(kwargs, ct::Reflect<T>::end());
                return kwargs;
            }

            template <class T, class... Args>
            struct CreateDataObject<T, ct::VariadicTypedef<Args...>>
            {
                static const int size = ct::Reflect<T>::N;
                static T* create(Args... args)
                {
                    T* obj = new T();
                    initDataMembers(*obj, args...);
                    return obj;
                }

                static boost::python::detail::keyword_range range()
                {
                    static std::array<boost::python::detail::keyword, ct::Reflect<T>::N + 1> s_keywords =
                        makeKeywordArgs<T>();
                    return std::make_pair<boost::python::detail::keyword const*, boost::python::detail::keyword const*>(
                        &s_keywords[0], &s_keywords[0] + ct::Reflect<T>::N + 1);
                }
            };

            template <class T, class BP>
            typename std::enable_if<ct::is_default_constructible<T>::value>::type addInit(BP& bpobj)
            {
                typedef CreateDataObject<
                    T,
                    typename FunctionSignatureBuilder<boost::python::object, ct::Reflect<T>::N>::VariadicTypedef_t>
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
            void extract(T& obj, const boost::python::object& bpobj, ct::Indexer<0>)
            {
                auto ptr = bpobj.ptr();
                boost::python::object python_member;
                if (PyObject_HasAttrString(ptr, ct::getName<0, T>()))
                {
                    python_member = bpobj.attr(ct::getName<0, T>());
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
                    convertFromPython(python_member, ct::set<0>(obj));
                }
            }

            template <class T, ct::index_t I>
            void extract(T& obj, const boost::python::object& bpobj, ct::Indexer<I> idx)
            {

                auto ptr = bpobj.ptr();
                boost::python::object python_member;
                if (PyObject_HasAttrString(ptr, ct::getName<I, T>()))
                {
                    python_member = bpobj.attr(ct::getName<I, T>());
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
                    convertFromPython(python_member, ct::set<I>(obj));
                }
                extract(obj, bpobj, --idx);
            }

            template <class T>
            ct::enable_if_not_reflected<T> pythonizeDataHelper(const char* /*name*/ = nullptr, const T* = nullptr)
            {
            }

            template <class T>
            ct::enable_if_reflected<T, std::string> repr(const T& data)
            {
                std::stringstream ss;
                ct::printStruct(ss, data);
                return ss.str();
            }

            template <class T>
            ct::enable_if_reflected<T, std::string> reprVec(const std::vector<T>& data)
            {
                std::stringstream ss;
                ss << '[';
                for (size_t i = 0; i < data.size(); ++i)
                {
                    if (i != 0)
                        ss << ", ";
                    ct::printStruct(ss, data[i]);
                }
                ss << ']';
                return ss.str();
            }

            template <class T>
            ct::enable_if_reflected<T> pythonizeDataHelper(const char* name = nullptr, const T* = nullptr)
            {
                static bool registered = false;
                if (registered)
                    return;
                boost::python::class_<T> bpobj(name ? name : mo::TypeTable::instance().typeToName(mo::TypeInfo(typeid(T))).c_str(),
                                               boost::python::no_init);
                addPropertyHelper<T>(bpobj, ct::Reflect<T>::end());
                addInit<T>(bpobj);
                bpobj.def("__repr__", &repr<T>);
                registered = true;
            }

            template <class T>
            ct::enable_if_reflected<T> pythonizeDataHelper(const char* name_ = nullptr, const std::vector<T>* = nullptr)
            {
                static bool registered = false;
                if (registered)
                    return;
                pythonizeDataHelper<T>(name_, static_cast<const T*>(nullptr));
                std::string name;
                if (name_)
                    name = name_;
                else
                    name = mo::TypeTable::instance().typeToName(mo::TypeInfo(typeid(T)));
                name += "List";
                boost::python::class_<std::vector<T>> bpobj(name.c_str());
                bpobj.def(boost::python::vector_indexing_suite<std::vector<T>>());
                bpobj.def("__repr__", &reprVec<T>);
                registered = true;
            }

            template <class K, class T>
            ct::enable_if_reflected<T> pythonizeDataHelper(const char* name = nullptr, const std::map<K, T>* = nullptr)
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
        inline auto convertFromPython(const boost::python::object& obj, T& result) -> ct::enable_if_not_reflected<T>
        {
            boost::python::extract<T> extractor(obj);
            result = extractor();
        }

        template <class T>
        inline auto convertFromPython(const boost::python::object& obj, T& result) -> ct::enable_if_reflected<T>
        {
            detail::extract(result, obj, ct::Reflect<T>::end());
        }

        inline void convertFromPython(const boost::python::object& obj, std::string& result)
        {
            result = boost::python::extract<std::string>(obj)();
        }

        template <class T>
        inline void convertFromPython(const boost::python::object& obj, ct::enable_if_reflected<T>& result)
        {
            detail::extract(result, obj, ct::Reflect<T>::end());
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
