#pragma once
#include <MetaObject/Parameters/IParameter.hpp>
#include "SerializationFunctionRegistry.hpp"
#include <MetaObject/Parameters/ITypedParameter.hpp>

#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/archives/json.hpp>
#include <functional>

namespace mo
{
    template<class T> class ITypedParameter;
    template<class T, int N, typename Enable> struct MetaParameter;
    namespace IO
    {
    namespace Cereal
    {   
        template<class T> struct Policy
        {
            Policy()
            {
                SerializationFunctionRegistry::Instance()->SetBinarySerializationFunctions(
                    TypeInfo(typeid(T)), 
                    std::bind(&Policy<T>::Serialize<cereal::BinaryOutputArchive>, std::placeholders::_1, std::placeholders::_2),
                    std::bind(&Policy<T>::DeSerialize<cereal::BinaryInputArchive>, std::placeholders::_1, std::placeholders::_2));

                SerializationFunctionRegistry::Instance()->SetXmlSerializationFunctions(
                    TypeInfo(typeid(T)), 
                    std::bind(&Policy<T>::Serialize<cereal::XMLOutputArchive>, std::placeholders::_1, std::placeholders::_2),
                    std::bind(&Policy<T>::DeSerialize<cereal::XMLInputArchive>, std::placeholders::_1, std::placeholders::_2));

                SerializationFunctionRegistry::Instance()->SetJsonSerializationFunctions(
                    TypeInfo(typeid(T)),
                    std::bind(&Policy<T>::Serialize<cereal::JSONOutputArchive>, std::placeholders::_1, std::placeholders::_2),
                    std::bind(&Policy<T>::DeSerialize<cereal::JSONInputArchive>, std::placeholders::_1, std::placeholders::_2));
            }

            template<class AR>
            static bool Serialize(IParameter* param, AR& ar)
            {
                ITypedParameter<T>* typed = dynamic_cast<ITypedParameter<T>*>(param);
                if(typed == nullptr)
                    return false;
                T* ptr = typed->GetDataPtr();
                if (ptr == nullptr)
                    return false;
                ar(cereal::make_nvp(param->GetName(), *ptr));
                return true;
            }
            template<class AR>
            static bool DeSerialize(IParameter* param, AR& ar)
            {
                ITypedParameter<T>* typed = dynamic_cast<ITypedParameter<T>*>(param);
                if (typed == nullptr)
                    return false;
                T* ptr = typed->GetDataPtr();
                if (ptr == nullptr)
                    return false;
                try
                {
                    ar(cereal::make_nvp(param->GetName(), *ptr));
                }catch(cereal::Exception& e)
                {
                    LOG(debug) << e.what();
                    return false;
                }
                typed->Commit();
                return true;
            }

        };
    } // namespace Cereal
    } // namespace IO
#define PARAMETER_CEREAL_SERIALIZATION_POLICY_INST_(N) \
    template<class T> struct MetaParameter<T, N, void>: public MetaParameter<T, N - 1, void> \
    { \
        static IO::Cereal::Policy<T> _cereal_policy;  \
        MetaParameter(const char* name): \
            MetaParameter<T, N-1, void>(name) \
        { \
            (void)&_cereal_policy; \
        } \
    }; \
    template<class T> IO::Cereal::Policy<T> MetaParameter<T, N, void>::_cereal_policy;

    PARAMETER_CEREAL_SERIALIZATION_POLICY_INST_(__COUNTER__)
}


