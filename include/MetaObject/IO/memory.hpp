#pragma once
#include "shared_ptr.hpp"
#include "Serializer.hpp"
#include "MetaObject/Parameters/IO/SerializationFunctionRegistry.hpp"
#include <MetaObject/Logging/Log.hpp>
#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/string.hpp>
namespace cereal
{
	template<class AR, class T> void save(AR& ar, rcc::shared_ptr<T> const & m)
	{
        mo::Serialize(ar, m.Get());
	}

	template<class AR, class T> void load(AR& ar, rcc::shared_ptr<T> & m)
	{
        if(m)
        {
            mo::DeSerialize(ar, m.Get());
        }
        else
        {
            std::string type;
            ar(make_nvp("TypeName", type));
            m = mo::MetaObjectFactory::Instance()->Create(type.c_str());
            mo::DeSerialize(ar, m.Get());
        }
	}
    template<class AR, class T> void save(AR& ar, rcc::weak_ptr<T> const & m)
    {
     
    }

    template<class AR, class T> void load(AR& ar, rcc::weak_ptr<T> & m)
    {
        
    }
}