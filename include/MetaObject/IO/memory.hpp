#pragma once
#include "shared_ptr.hpp"
#include "MetaObject/IO/Serializer.hpp
#include <cereal/cereal.hpp>
namespace cereal
{
	template<class Archive, class T> void save(Archive& ar, rcc::shared_ptr<T> const & m)
	{
		ar(cereal::make_nvp("ObjectType", std::string(m->GetTypeName())));
		mo::SerializerFactory::
	}

	template<class Archive, class T> void load(Archive& ar, rcc::shared_ptr<T> & m)
	{

	}
}