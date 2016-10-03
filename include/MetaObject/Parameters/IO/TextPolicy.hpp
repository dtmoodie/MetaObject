#pragma once
#include <MetaObject/Parameters/IParameter.hpp>
#include "SerializationFunctionRegistry.hpp"

namespace mo 
{ 
namespace IO 
{ 
namespace Text 
{
    template<typename T> bool Serialize(IParameter* param, std::stringstream& ss)
    {
        ITypedParameter<T>* typed = dynamic_cast<ITypedParameter<T>*>(param);
        if (typed)
        {
            ss << typed->GetData();
        }
    }

    template<typename T> bool DeSerialize(IParameter* param, std::stringstream& ss)
    {
        ITypedParameter<T>* typed = dynamic_cast<ITypedParameter<T>*>(param);
        if (typed)
        {
            auto ptr = typed->GetDataPtr();
            if (ptr)
            {
                ss >> *ptr;
            }
        }
    }
    template<class T> struct Policy
    {
        Policy()
        {
            SerializationFunctionRegistry::Instance()->SetTextSerializationFunctions(
                TypeInfo(typeid(T)),
                std::bind(&Serialize, std::placeholders_1, std::placeholders_2),
                std::bind(&DeSerialize, std::placeholders_1, std::placeholders_2));
        }
    };
} // Text 
} // IO

#define PARAMETER_SERIALIZATION_POLICY_INST_(N) \
  template<class T> struct MetaParameter<T, N>: public MetaParameter<T, N - 1, void>, public IO::Text::Policy<T> \
    { \
        MetaParameter(const char* name): \
            MetaParameter<T, N-1, void>(name){} \
    };

PARAMETER_SERIALIZATION_POLICY_INST_(__COUNTER__)
} // mo