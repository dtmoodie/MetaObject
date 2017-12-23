#pragma once
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/params/IParam.hpp"
#include "MetaObject/params/ITAccessibleParam.hpp"
#include <boost/python.hpp>

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
            void registerConverters(const mo::TypeInfo& type, Set_t&& setter, const Get_t&& getter);
            Set_t getSetter(const mo::TypeInfo& type);
            Get_t getGetter(const mo::TypeInfo& type);
            std::vector<mo::TypeInfo> listConverters();
        private:
            std::map<mo::TypeInfo, std::pair<Set_t, Get_t>> m_registered_converters;
        };

        template<class T>
        struct DataConverter
        {
            DataConverter()
            {
                DataConverterRegistry::instance()->registerConverters(mo::TypeInfo(typeid(T)), std::bind(&DataConverter<T>::set, _1, _2), std::bind(&DataConverter<T>::get, _1, _2));
            }

            bool set(mo::ParamBase* param, const boost::python::object& obj)
            {
                if(param->getTypeInfo() == mo::TypeInfo(typeid(T)))
                {
                    auto extractor = boost::python::extract<T>(obj);
                    if(extractor.check())
                    {
                        if (auto typed = static_cast<mo::ITAccessibleParam<T>*>(param))
                        {
                            auto token = typed->access();
                            token() = extractor();
                            return true;
                        }
                    }
                }
                return false;
            }

            boost::python::object get(const mo::ParamBase* param)
            {
                if(param->getTypeInfo() == mo::TypeInfo(typeid(T)))
                {
                    if(auto typed = static_cast<const mo::ITAccessibleParam<T>*>(param))
                    {
                        auto token = typed->access();
                        return token();
                    }
                }
                return {};
            }
        };
    }
}
