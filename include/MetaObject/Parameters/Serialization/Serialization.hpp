#pragma once
#include "MetaObject/Detail/Export.hpp"

namespace cereal
{
    class BinaryOutputArchive;
    class BinaryInoutArchive;
}
namespace mo
{
    class IMetaObject;
    class IParameter;
    namespace Serialization
    {
        MO_EXPORTS void Serialize(IMetaObject& obj, cereal::BinaryOutputArchive& ar);
        MO_EXPORTS void Deserialize(IMetaObject& obj, cereal::BinaryInoutArchive& ar);
        MO_EXPORTS IMetaObject* Deserialize(cereal::BinaryInoutArchive& ar);
    }
}