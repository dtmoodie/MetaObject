#pragma once
#include "RuntimeObjectSystem/shared_ptr.hpp"
#include "Serializer.hpp"
#include "MetaObject/serialization//SerializationFactory.hpp"
#include <MetaObject/logging/Log.hpp>
#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/string.hpp>
namespace cereal {
template<class AR, class T> void save(AR& ar, rcc::shared_ptr<T> const & m) {
    if(mo::CheckHasBeenSerialized(m->GetObjectId())) {
        std::string type = m->GetTypeName();
        ObjectId id = m->GetObjectId();
        ar(cereal::make_nvp("TypeId", id.m_ConstructorId));
        ar(cereal::make_nvp("InstanceId", id.m_PerTypeId));
        ar(make_nvp("TypeName", type));
        return;
    }
    mo::Serialize(ar, m.Get());
    mo::SetHasBeenSerialized(m->GetObjectId());
}

template<class AR, class T> void load(AR& ar, rcc::shared_ptr<T> & m) {
    if(m) {
        if (mo::CheckHasBeenSerialized(m->GetObjectId()))
            return;
        mo::DeSerialize(ar, m.Get());
    } else {
        std::string type;
        ObjectId id;
        ar(cereal::make_nvp("TypeId", id.m_ConstructorId));
        ar(cereal::make_nvp("InstanceId", id.m_PerTypeId));
        ar(make_nvp("TypeName", type));
        if(auto obj = mo::MetaObjectFactory::instance()->get(id, type.c_str())) {
            m = obj;
        } else {
            m = mo::MetaObjectFactory::instance()->create(type.c_str());
        }
        if (mo::CheckHasBeenSerialized(m->GetObjectId()))
            return;
        mo::DeSerialize(ar, m.Get());
    }
    mo::SetHasBeenSerialized(m->GetObjectId());
}

template<class AR, class T> void save(AR& ar, rcc::weak_ptr<T> const & m) {
    std::string type = m->GetTypeName();
    ObjectId id = m->GetObjectId();
    ar(cereal::make_nvp("TypeId", id.m_ConstructorId));
    ar(cereal::make_nvp("InstanceId", id.m_PerTypeId));
    ar(make_nvp("TypeName", type));
}

template<class AR, class T> void load(AR& ar, rcc::weak_ptr<T> & m) {
    std::string type;
    ObjectId id;
    ar(cereal::make_nvp("TypeId", id.m_ConstructorId));
    ar(cereal::make_nvp("InstanceId", id.m_PerTypeId));
    ar(make_nvp("TypeName", type));
    m = mo::MetaObjectFactory::instance()->get(id, type.c_str());
}
}
