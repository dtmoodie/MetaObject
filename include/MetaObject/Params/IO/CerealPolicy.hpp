#pragma once
#include <MetaObject/Params/IParam.hpp>
#include "SerializationFactory.hpp"
#include <MetaObject/Params/ITParam.hpp>

#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/archives/json.hpp>
#include <functional>

namespace mo {
template<class T> class ITParam;
template<class T, int N, typename Enable> struct MetaParam;
namespace IO {
namespace Cereal {
template<class T> struct Policy {
    Policy() {
        SerializationFactory::Instance()->SetBinarySerializationFunctions(
            TypeInfo(typeid(T)),
            std::bind(&Policy<T>::Serialize<cereal::BinaryOutputArchive>, std::placeholders::_1, std::placeholders::_2),
            std::bind(&Policy<T>::DeSerialize<cereal::BinaryInputArchive>, std::placeholders::_1, std::placeholders::_2));

        SerializationFactory::Instance()->SetXmlSerializationFunctions(
            TypeInfo(typeid(T)),
            std::bind(&Policy<T>::Serialize<cereal::XMLOutputArchive>, std::placeholders::_1, std::placeholders::_2),
            std::bind(&Policy<T>::DeSerialize<cereal::XMLInputArchive>, std::placeholders::_1, std::placeholders::_2));

        SerializationFactory::Instance()->SetJsonSerializationFunctions(
            TypeInfo(typeid(T)),
            std::bind(&Policy<T>::Serialize<cereal::JSONOutputArchive>, std::placeholders::_1, std::placeholders::_2),
            std::bind(&Policy<T>::DeSerialize<cereal::JSONInputArchive>, std::placeholders::_1, std::placeholders::_2));
    }

    template<class AR>
    static bool Serialize(IParam* param, AR& ar) {
        ITParam<T>* typed = dynamic_cast<ITParam<T>*>(param);
        if(typed == nullptr)
            return false;
        T* ptr = typed->GetDataPtr();
        if (ptr == nullptr)
            return false;
        ar(cereal::make_nvp(param->getName(), *ptr));
        return true;
    }
    template<class AR>
    static bool DeSerialize(IParam* param, AR& ar) {
        ITParam<T>* typed = dynamic_cast<ITParam<T>*>(param);
        if (typed == nullptr)
            return false;
        T* ptr = typed->GetDataPtr();
        if (ptr == nullptr)
            return false;
        auto nvp = cereal::make_optional_nvp(param->getName(), *ptr, *ptr);
        try {
            ar(nvp);
        } catch(...) {
            return false;
        }

        if(nvp.success) {
            typed->commit();
            return true;
        }
        return false;
    }

};
} // namespace Cereal
} // namespace IO
/*template<class T> using DetectSerializer = typename std::enable_if<
    cereal::traits::detail::count_input_serializers<T, cereal::JSONInputArchive>::value != 0 &&
    cereal::traits::detail::count_input_serializers<T, cereal::XMLInputArchive>::value != 0 &&
    cereal::traits::detail::count_input_serializers<T, cereal::BinaryInputArchive>::value != 0
        >::type;*/
template<class T> using DetectSerializer = void;

#define Param_CEREAL_SERIALIZATION_POLICY_INST_(N) \
    template<class T> struct MetaParam<T, N, \
        DetectSerializer<T>>: public MetaParam<T, N - 1, void> \
    { \
        static IO::Cereal::Policy<T> _cereal_policy;  \
        MetaParam(const char* name): \
            MetaParam<T, N-1, void>(name) \
        { \
            (void)&_cereal_policy; \
        } \
    }; \
    template<class T> IO::Cereal::Policy<T> MetaParam<T, N, DetectSerializer<T>>::_cereal_policy;

Param_CEREAL_SERIALIZATION_POLICY_INST_(__COUNTER__)
}

