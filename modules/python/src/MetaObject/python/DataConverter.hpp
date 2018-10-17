#pragma once
#include "MetaObject/core/Demangle.hpp"
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/params/AccessToken.hpp"
#include "MetaObject/params/IParam.hpp"
#include "MetaObject/params/ITAccessibleParam.hpp"
#include "MetaObject/params/OutputParam.hpp"
#include "MetaObject/python/PythonSetup.hpp"

#include "converters.hpp"

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
            }

            static bool set(mo::ParamBase* param, const boost::python::object& obj)
            {
                if (param->getTypeInfo() == TypeInfo(typeid(T)))
                {
                    if (auto typed = dynamic_cast<TParam<T>*>(param))
                    {
                        auto token = typed->access();
                        mo::python::convertFromPython(obj, token());
                        return true;
                    }
                }
                return false;
            }

            static boost::python::object get(const ParamBase* param)
            {
                if (param->getTypeInfo() == mo::TypeInfo(typeid(T)))
                {
                    if (param->checkFlags(ParamFlags::Output_e))
                    {
                        if (auto output_param = dynamic_cast<const OutputParam*>(param))
                        {
                            param = output_param->getOutputParam();
                        }
                    }
                    if (auto typed = dynamic_cast<const TParam<T>*>(param))
                    {
                        if (typed->isValid())
                        {
                            auto token = typed->read();
                            return convertToPython(token());
                        }
                    }
                    else
                    {

                        MO_LOG(debug) << "Failed to cast parameter (" << mo::Demangle::typeToName(param->getTypeInfo())
                                      << ") to the correct type for " << mo::Demangle::typeToName(TypeInfo(typeid(T)));
                    }
                }
                else
                {
                    MO_LOG(trace) << "Incorrect datatype input " << mo::Demangle::typeToName(param->getTypeInfo())
                                  << " expcted " << mo::Demangle::typeToName(mo::TypeInfo(typeid(T)));
                }
                return {};
            }
        };
    }
}
