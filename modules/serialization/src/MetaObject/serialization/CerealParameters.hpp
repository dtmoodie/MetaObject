#pragma once
#include "MetaObject/logging/logging.hpp"
#include "MetaObject/params/IParam.hpp"
#include "SerializationFactory.hpp"
#include "cereal/archives/json.hpp"

namespace cereal
{
    template <class AR>
    void load(AR& ar, std::vector<mo::IParam*>& Params)
    {
        for (auto& param : Params)
        {
            if (param->checkFlags(mo::ParamFlags::Output_e) || param->checkFlags(mo::ParamFlags::Input_e))
                continue;
            auto func1 = mo::SerializationFactory::instance()->getLoadFunction(param->getTypeInfo(), ar);
            if (func1)
            {
                if (!func1(param, ar))
                {
                    MO_LOG(debug, "Unable to deserialize {} of type {}", param->getName(),param->getTypeInfo().name());
                }
            }
            else
            {
                MO_LOG(debug,  "No deserialization function exists for {} of type {}",param->getName(),
                              param->getTypeInfo().name());
            }
        }
    }
    template <class AR>
    void save(AR& ar, std::vector<mo::IParam*> const& Params)
    {
        for (auto& param : Params)
        {
            if (param->checkFlags(mo::ParamFlags::Output_e) || param->checkFlags(mo::ParamFlags::Input_e))
                continue;
            auto func1 = mo::SerializationFactory::instance()->getSaveFunction(param->getTypeInfo(), ar);
            if (func1)
            {
                if (!func1(param, ar))
                {
                    MO_LOG(debug, "Unable to deserialize {} of type {}",param->getName(), param->getTypeInfo().name());
                }
            }
            else
            {
                MO_LOG(debug, "No serialization function exists for {} of type ", param->getName(),param->getTypeInfo().name());
            }
        }
    }
}
