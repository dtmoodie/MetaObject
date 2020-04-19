#ifndef MO_PYTHON_DATACONVERTER_HPP
#define MO_PYTHON_DATACONVERTER_HPP
#include "converters.hpp"

#include <MetaObject/core/TypeTable.hpp>
#include <MetaObject/detail/Export.hpp>
#include <MetaObject/params/AccessToken.hpp>
#include <MetaObject/params/IControlParam.hpp>
#include <MetaObject/params/IPublisher.hpp>
#include <MetaObject/params/ITControlParam.hpp>
#include <MetaObject/python/PythonSetup.hpp>

#include <boost/python.hpp>
#include <boost/python/extract.hpp>

#include <ct/interop/boost_python/PythonConverter.hpp>

#include <map>

namespace mo
{
    namespace python
    {
        struct MO_EXPORTS DataConverterRegistry
        {
            using Set_t = std::function<bool(mo::IControlParam*, const boost::python::object&)>;
            using Get_t = std::function<boost::python::object(const mo::IControlParam*)>;

            static DataConverterRegistry* instance();
            static DataConverterRegistry* instance(SystemTable* table);
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
            ParamConverter(SystemTable* table)
            {
                DataConverterRegistry::instance(table)->registerConverters(
                    mo::TypeInfo(typeid(T)),
                    std::bind(&ParamConverter<T>::set, std::placeholders::_1, std::placeholders::_2),
                    std::bind(&ParamConverter<T>::get, std::placeholders::_1));
            }

            static bool set(mo::IParam* param, const boost::python::object& obj)
            {
                if (param->checkFlags(mo::ParamFlags::kCONTROL))
                {
                    auto control = dynamic_cast<IControlParam*>(param);
                    if (control->getTypeInfo() == TypeInfo(typeid(T)))
                    {
                        if (auto typed = dynamic_cast<ITControlParam<T>*>(control))
                        {
                            T value = typed->getValue();
                            if (ct::convertFromPython(obj, value))
                            {
                                typed->setValue(std::move(value));
                            }
                        }
                    }
                }

                return false;
            }

            static boost::python::object get(const IParam* param)
            {
                static const mo::TypeInfo type = mo::TypeInfo::create<T>();
                if (param->checkFlags(mo::ParamFlags::kCONTROL))
                {
                    auto typed = dynamic_cast<const ITControlParam<T>*>(param);
                    if (typed)
                    {
                        return ct::convertToPython(typed->getValue());
                    }
                }

                /*if (param->checkFlags(ParamFlags::kOUTPUT))
                {
                    if (auto output_param = dynamic_cast<const IPublisher*>(param))
                    {
                        param = output_param->getPublisher();
                    }
                }*/

                /*if (auto typed = dynamic_cast<const TParam<T>*>(param))
                {
                    if (typed->isValid())
                    {
                        auto token = typed->read();
                        return ct::convertToPython(token());
                    }
                }
                else
                {

                    MO_LOG(debug,
                           "Failed to cast parameter ({}) to the correct type for {}",
                           mo::TypeTable::instance()->typeToName(param->getTypeInfo()),
                           mo::TypeTable::instance()->typeToName(TypeInfo(typeid(T))));
                }*/

                return {};
            }
        };
    } // namespace python
} // namespace mo

#endif // MO_PYTHON_DATACONVERTER_HPP