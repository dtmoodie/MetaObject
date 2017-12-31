#pragma once
#include "MetaObject/core/Demangle.hpp"
#include "MetaObject/params/reflect_data.hpp"
#include "MetaObject/python/converters.hpp"
#include "boost/python.hpp"
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

namespace mo
{
    namespace python
    {

        template <class T, class Enable>
        struct FromPythonDataConverter;

        template <class T, class Enable>
        struct ToPythonDataConverter;

        namespace detail
        {
            template <class T, class BP>
            void addPropertyHelper(BP& bpobj, mo::_counter_<0>)
            {
                using Return_t = std::decay_t<decltype(mo::reflect::get<0, T>(std::declval<T const>()))>;
                bpobj.add_property(mo::reflect::getName<0, T>(),
                                   static_cast<Return_t (*)(const T&)>(&mo::reflect::getValue<0, T>));
            }

            template <class T, int I, class BP>
            void addPropertyHelper(BP& bpobj, mo::_counter_<I>)
            {
                using Return_t = std::decay_t<decltype(mo::reflect::get<I, T>(std::declval<T const>()))>;
                bpobj.add_property(mo::reflect::getName<I, T>(),
                                   static_cast<Return_t (*)(const T&)>(&mo::reflect::getValue<I, T>));
                addPropertyHelper<T>(bpobj, mo::_counter_<I - 1>());
            }

            template <class T>
            void extract(T& obj, const boost::python::object& bpobj, mo::_counter_<0>)
            {
                auto ptr = bpobj.ptr();
                using DType = std::decay_t<decltype(mo::reflect::get<0>(obj))>;
                boost::python::object python_member;
                if (PyObject_HasAttrString(ptr, mo::reflect::getName<0, T>()))
                {
                    python_member = bpobj.attr(mo::reflect::getName<0, T>());
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
                    if (mo::reflect::ReflectData<DType>::IS_SPECIALIZED)
                    {
                        mo::python::FromPythonDataConverter<DType, void> cvt;
                        mo::reflect::get<0>(obj) = cvt(python_member);
                    }
                    else
                    {
                        boost::python::extract<DType> ext(python_member);
                        mo::reflect::get<0>(obj) = ext();
                    }
                }
            }

            template <class T, int I>
            void extract(T& obj, const boost::python::object& bpobj, mo::_counter_<I>)
            {
                auto ptr = bpobj.ptr();
                using DType = std::decay_t<decltype(mo::reflect::get<I>(obj))>;
                boost::python::object python_member;
                if (PyObject_HasAttrString(ptr, mo::reflect::getName<I, T>()))
                {
                    python_member = bpobj.attr(mo::reflect::getName<I, T>());
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
                    if (mo::reflect::ReflectData<DType>::IS_SPECIALIZED)
                    {
                        FromPythonDataConverter<DType, void> cvt;
                        mo::reflect::get<I>(obj) = cvt(python_member);
                    }
                    else
                    {
                        boost::python::extract<DType> ext(python_member);
                        mo::reflect::get<I>(obj) = ext();
                    }
                }
                extract(obj, bpobj, mo::_counter_<I - 1>());
            }

            template <class T>
            mo::reflect::enable_if_not_reflected<T> pythonizeDataHelper(const char* name = nullptr, const T* = nullptr)
            {
            }

            template <class T>
            mo::reflect::enable_if_reflected<T, std::string> repr(const T& data)
            {
                std::stringstream ss;
                mo::reflect::printStruct(ss, data);
                return ss.str();
            }

            template <class T>
            mo::reflect::enable_if_reflected<T, std::string> reprVec(const std::vector<T>& data)
            {
                std::stringstream ss;
                ss << '[';
                for (size_t i = 0; i < data.size(); ++i)
                {
                    if (i != 0)
                        ss << ", ";
                    mo::reflect::printStruct(ss, data[i]);
                }
                ss << ']';
                return ss.str();
            }

            template <class T>
            mo::reflect::enable_if_reflected<T> pythonizeDataHelper(const char* name = nullptr, const T* = nullptr)
            {
                boost::python::class_<T> bpobj(name ? name : mo::Demangle::typeToName(mo::TypeInfo(typeid(T))).c_str());
                detail::addPropertyHelper<T>(bpobj, mo::_counter_<mo::reflect::ReflectData<T>::N - 1>());
                bpobj.def("__repr__", &repr<T>);
            }

            template <class T>
            mo::reflect::enable_if_reflected<T> pythonizeDataHelper(const char* name_ = nullptr,
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
            mo::reflect::enable_if_reflected<T> pythonizeDataHelper(const char* name = nullptr,
                                                                    const std::map<K, T>* = nullptr)
            {
                pythonizeDataHelper<T>(name, static_cast<const T*>(nullptr));
                boost::python::class_<std::vector<T>> bpobj((std::string(name) + "Map").c_str());
                bpobj.def(boost::python::map_indexing_suite<std::map<K, T>>());
            }

            template <class T>
            void pythonizeData(const char* name)
            {
                detail::pythonizeDataHelper(name, static_cast<const T*>(nullptr));
            }
        } // namespace mo::python::detail
    }     // namespace mo::python
} // namespace mo
