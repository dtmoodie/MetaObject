#pragma once
#include "lambda.hpp"
#include <boost/python.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>

namespace cereal
{
    // For python -> C++
    struct PythonInputArchive: public InputArchive<PythonInputArchive>
    {
        PythonInputArchive(const boost::python::object & obj):
            InputArchive<PythonInputArchive>(this),
            input_object(obj)
        {
            current_object = &input_object;
        }

        template<class T>
        void loadValue(T& value)
        {
            boost::python::extract<T> extractor(*current_object);
            if(extractor.check())
            {
                value = extractor();
            }
        }

        const boost::python::object* current_object;
        const boost::python::object& input_object;
        size_t current_index = 0;
    };

    template<class T> inline
    void load(PythonInputArchive& ar, cereal::NameValuePair<T>& value)
    {
        auto prev = ar.current_object;
        auto prev_index = ar.current_index;
        if(PyObject_HasAttrString(prev->ptr(), value.name))
        {
            boost::python::object attr = ar.current_object->attr(value.name);
            ar.current_object = &attr;
            ar(value.value);
        }else
        {
            const size_t len = boost::python::len(*ar.current_object);
            if(ar.current_index < len)
            {
                boost::python::object obj((*ar.current_object)[ar.current_index]);
                ar.current_object = &obj;
                ar(value.value);
            }
        }
        ar.current_object = prev;
        ar.current_index++;
    }

    template<class T> inline
    void load(PythonInputArchive& ar, std::vector<T>& value)
    {
        auto prev = ar.current_object;
        boost::python::list list(*ar.current_object);
        const auto len = boost::python::len(list);
        value.resize(len);
        for(size_t i = 0; i < len; ++i)
        {
            boost::python::object item = list[i];
            ar.current_object = &item;
            ar(value[i]);
        }
        ar.current_object = prev;
    }

    template<class T> inline
    typename std::enable_if<std::is_arithmetic<T>::value, void>::type
    load(PythonInputArchive & ar, T & t)
    {
        ar.loadValue(t);
    }

    inline
    void load(PythonInputArchive & ar, std::string & t)
    {
        ar.loadValue(t);
    }

    template<class K, class T> inline
    void load(PythonInputArchive& ar, std::map<K, T>& map)
    {
        // TODO
    }
}
