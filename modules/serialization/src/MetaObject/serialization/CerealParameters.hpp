#pragma once
#include "cereal/archives/json.hpp"
#include "MetaObject/params/IParam.hpp"
#include "MetaObject/logging/logging.hpp"
#include "SerializationFactory.hpp"

namespace cereal {
template<class AR>
void load(AR& ar, std::vector<mo::IParam*>& Params) {
    for (auto& param : Params) {
        if (param->checkFlags(mo::ParamFlags::Output_e) || param->checkFlags(mo::ParamFlags::Input_e))
            continue;
        auto func1 = mo::SerializationFactory::instance()->getLoadFunction(param->getTypeInfo(), ar);
        if (func1) {
            if (!func1(param, ar)) {
                MO_LOG(debug) << "Unable to deserialize " << param->getName() << " of type " << param->getTypeInfo().name();
            }
        } else {
            MO_LOG(debug) << "No deserialization function exists for  " << param->getName() << " of type " << param->getTypeInfo().name();
        }
    }
}
template<class AR>
void save(AR& ar, std::vector<mo::IParam*> const& Params) {
    for (auto& param : Params) {
        if (param->checkFlags(mo::ParamFlags::Output_e) || param->checkFlags(mo::ParamFlags::Input_e))
            continue;
        auto func1 = mo::SerializationFactory::instance()->getSaveFunction(param->getTypeInfo(), ar);
        if (func1) {
            if (!func1(param, ar)) {
                MO_LOG(debug) << "Unable to deserialize " << param->getName() << " of type " << param->getTypeInfo().name();
            }
        } else {
            MO_LOG(debug) << "No serialization function exists for  " << param->getName() << " of type " << param->getTypeInfo().name();
        }
    }
}

}
