#pragma once
#include "cereal/archives/json.hpp"
#include "MetaObject/Params/IParam.hpp"
#include "MetaObject/Logging/Log.hpp"
#include "SerializationFactory.hpp"

namespace cereal {
template<class AR>
void load(AR& ar, std::vector<mo::IParam*>& Params) {
    for (auto& param : Params) {
        if (param->checkFlags(mo::Output_e) || param->checkFlags(mo::Input_e))
            continue;
        auto func1 = mo::SerializationFactory::Instance()->GetLoadFunction(param->getTypeInfo(), ar);
        if (func1) {
            if (!func1(param, ar)) {
                LOG(debug) << "Unable to deserialize " << param->getName() << " of type " << param->getTypeInfo().name();
            }
        } else {
            LOG(debug) << "No deserialization function exists for  " << param->getName() << " of type " << param->getTypeInfo().name();
        }
    }
}
template<class AR>
void save(AR& ar, std::vector<mo::IParam*> const& Params) {
    for (auto& param : Params) {
        if (param->checkFlags(mo::Output_e) || param->checkFlags(mo::Input_e))
            continue;
        auto func1 = mo::SerializationFactory::Instance()->GetSaveFunction(param->getTypeInfo(), ar);
        if (func1) {
            if (!func1(param, ar)) {
                LOG(debug) << "Unable to deserialize " << param->getName() << " of type " << param->getTypeInfo().name();
            }
        } else {
            LOG(debug) << "No serialization function exists for  " << param->getName() << " of type " << param->getTypeInfo().name();
        }
    }
}

}