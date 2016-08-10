#include "MetaObject/Parameters/Serialization/Serialization.hpp"
//#include <cereal/cereal.hpp>

using namespace mo;
using namespace mo::Serialization;
using namespace cereal;


void Serialization::Serialize(IMetaObject& obj, BinaryOutputArchive& ar)
{

}

void Serialization::Deserialize(IMetaObject& obj, BinaryInoutArchive& ar)
{

}

IMetaObject* Serialization::Deserialize(BinaryInoutArchive& ar)
{
    return nullptr;
}