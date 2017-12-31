#pragma once
#include "CerealParameters.hpp"
#include "MetaObject/logging/logging.hpp"
#include "MetaObject/object/IMetaObject.hpp"
#include "MetaObject/object/MetaObjectFactory.hpp"
#include "RuntimeObjectSystem/shared_ptr.hpp"

#include <type_traits>

namespace cereal
{
    template <class AR, class T>
    typename std::enable_if<std::is_base_of<mo::IMetaObject, T>::value>::type load(AR& ar, rcc::shared_ptr<T>& obj)
    {
        std::string type;
        ar(CEREAL_NVP(type));
        obj = mo::MetaObjectFactory::instance()->create(type.c_str());
        if (!obj)
        {
            MO_LOG(warning) << "Unable to create object with type: " << type;
            return;
        }
        auto Params = obj->getParams();
        ar(CEREAL_NVP(Params));
    }

    template <class AR, class T>
    typename std::enable_if<std::is_base_of<mo::IMetaObject, T>::value>::type save(AR& ar,
                                                                                   rcc::shared_ptr<T> const& obj)
    {
        if (obj)
        {
            auto Params = obj->getParams();
            std::string type = obj->GetTypeName();
            ar(CEREAL_NVP(type));
            ar(CEREAL_NVP(Params));
        }
    }
}
