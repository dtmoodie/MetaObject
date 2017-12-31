#pragma once
#include "MetaObject/core/Demangle.hpp"
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/params/IParam.hpp"
#include "MetaObject/params/ITAccessibleParam.hpp"
#include "MetaObject/python/PythonSetup.hpp"
#include "converters.hpp"

#include "CerealPythonArchive.hpp"

#include <boost/python.hpp>
#include <boost/python/extract.hpp>

#include <map>

namespace mo
{
    namespace python
    {
        struct MO_EXPORTS DataConverterRegistry
        {
            typedef std::function<bool(mo::ParamBase*, const boost::python::object&)> Set_t;
            typedef std::function<boost::python::object(const mo::ParamBase*)> Get_t;

            static DataConverterRegistry* instance();
            void registerConverters(const mo::TypeInfo& type, const Set_t& setter, const Get_t& getter);
            Set_t getSetter(const mo::TypeInfo& type);
            Get_t getGetter(const mo::TypeInfo& type);
            std::vector<mo::TypeInfo> listConverters();

          private:
            std::map<mo::TypeInfo, std::pair<Set_t, Get_t>> m_registered_converters;
        };

        template <class T>
        struct ParamConverter
        {
            ParamConverter()
            {
                DataConverterRegistry::instance()->registerConverters(
                    mo::TypeInfo(typeid(T)),
                    std::bind(&ParamConverter<T>::set, std::placeholders::_1, std::placeholders::_2),
                    std::bind(&ParamConverter<T>::get, std::placeholders::_1));
                // registerSetupFunction(std::bind(&ParamConverter<T>::setupStruct));
            }

            static bool set(mo::ParamBase* param, const boost::python::object& obj)
            {
                if (param->getTypeInfo() == mo::TypeInfo(typeid(T)))
                {
                    if (auto typed = dynamic_cast<mo::ITAccessibleParam<T>*>(param))
                    {
                        auto token = typed->access();
                        mo::python::convertFromPython(token(), obj);
                        return true;
                    }
                }
                return false;
            }

            static boost::python::object get(const mo::ParamBase* param)
            {
                if (param->getTypeInfo() == mo::TypeInfo(typeid(T)))
                {
                    if (auto typed = dynamic_cast<const mo::ITAccessibleParam<T>*>(param))
                    {
                        auto token = typed->access();
                        return boost::python::object(token());
                    }
                }
                return {};
            }
        };
    }
}
