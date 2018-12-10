#pragma once
#include "MetaObject/serialization/SerializationFactory.hpp"
#include <MetaObject/logging/logging.hpp>
#include <MetaObject/object/IMetaObject.hpp>
#include <MetaObject/object/MetaObjectFactory.hpp>
#include <MetaObject/params/IParam.hpp>
#include <RuntimeObjectSystem/IObject.h>

#include "RuntimeObjectSystem/shared_ptr.hpp"

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

namespace cereal
{
    /*template <class AR>
    void save(AR& ar, const mo::IParam* param)
    {
        auto serializer = mo::SerializationFactory::instance()->getSerializer<AR>(param->getTypeInfo());
        if (serializer)
        {
            serializer(param, ar);
        }
    }*/

    template <class AR>
    void save(AR& ar, const mo::IMetaObject& obj)
    {
        auto params = obj.getParams();
        ar(cereal::make_nvp("parameters", params));
    }

    template <class AR>
    void load(AR& ar, mo::IMetaObject& obj)
    {
        auto params = obj.getParams();
        ar(cereal::make_nvp("parameters", params));
    }

    template <class AR, class T>
    void save(AR& ar, rcc::shared_ptr<T> const& m)
    {
        if (m != nullptr)
        {
            const std::string type = m->GetTypeName();
            const auto id = m->GetPerTypeId();
            ar(cereal::make_nvp("type", type));
            ar(cereal::make_nvp("id", id));
            ar(cereal::make_nvp("object", *m));
        }
    }

    template <class AR, class T>
    void load(AR& ar, rcc::shared_ptr<T>& m)
    {
        if (!m)
        {
            std::string type;
            PerTypeObjectId id;
            ar(cereal::make_nvp("type", type));
            ar(cereal::make_nvp("id", id));
            auto ctr = mo::MetaObjectFactory::instance()->getConstructor(type.c_str());
            MO_ASSERT(ctr != nullptr);
            auto obj = ctr->GetConstructedObject(id);
            if (obj == nullptr)
            {
                obj = ctr->Construct();
                obj->Init(true);
            }
            m = obj;
        }

        ar(cereal::make_nvp("object", *m));
    }

    template <class AR, class T>
    void save(AR& ar, rcc::weak_ptr<T> const& m)
    {
    }

    template <class AR, class T>
    void load(AR& ar, rcc::weak_ptr<T>& m)
    {
    }
}
