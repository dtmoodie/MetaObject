#pragma once
#include "PythonSetup.hpp"
#include "detail/converters.hpp"
#include <MetaObject/logging/logging.hpp>

namespace mo
{
    namespace python
    {
        template <class T, class Enable = void>
        struct FromPythonDataConverter
        {
        };

        template <class T, class Enable = void>
        struct ToPythonDataConverter
        {
        };

        template <class T>
        struct ToPythonDataConverter<T, mo::reflect::enable_if_reflected<T>>
        {
            ToPythonDataConverter()
            {
                python::registerSetupFunction(std::bind(&python::detail::pythonizeData<T>, nullptr));
            }
        };

        template <class T>
        struct ToPythonDataConverter<std::vector<T>, mo::reflect::enable_if_reflected<T>>
        {
            ToPythonDataConverter()
            {
                python::registerSetupFunction(std::bind(&python::detail::pythonizeData<std::vector<T>>, nullptr));
            }
        };

        template <>
        struct FromPythonDataConverter<std::string, void>
        {
            void operator()(std::string& result, const boost::python::object& obj)
            {
                boost::python::extract<std::string> extractor(obj);
                result = extractor();
            }
        };

        template <class K, class T>
        struct FromPythonDataConverter<std::map<K, T>, void>
        {
            void operator()(std::map<K, T>& result, const boost::python::object& obj)
            {
                boost::python::extract<std::map<K, T>> extractor(obj);
                result = extractor();
            }
        };

        template <class T>
        struct FromPythonDataConverter<
            T,
            std::enable_if_t<!mo::reflect::ReflectData<T>::IS_SPECIALIZED && !mo::reflect::is_container<T>::value>>
        {
            T operator()(const boost::python::object& obj)
            {
                T result;
                (*this)(result, obj);
                return result;
            }
            void operator()(T& result, const boost::python::object& obj)
            {
                boost::python::extract<T> extractor(obj);
                result = extractor();
            }
        };

        // read a python object into a c++ struct
        template <class T>
        struct FromPythonDataConverter<T, mo::reflect::enable_if_reflected<T>>
        {
            void operator()(T& result, const boost::python::object& obj)
            {
                detail::extract(result, obj, mo::_counter_<mo::reflect::ReflectData<T>::N - 1>());
            }

            T operator()(const boost::python::object& obj)
            {
                T result;
                (*this)(result, obj);
                return result;
            }
        };

        template <class T>
        struct FromPythonDataConverter<std::vector<T>, void>
        {
            void operator()(std::vector<T>& result, const boost::python::object& obj)
            {
                const ssize_t len = boost::python::len(obj);
                result.resize(len);
                for (ssize_t i = 0; i < len; ++i)
                {
                    FromPythonDataConverter<T, void> cvt;
                    cvt(result[i], boost::python::object(obj[i]));
                }
            }

            std::vector<T> operator()(const boost::python::object& obj)
            {
                std::vector<T> result;
                (*this)(result, obj);
                return result;
            }
        };
    }
}

#include "detail/converters.hpp"
