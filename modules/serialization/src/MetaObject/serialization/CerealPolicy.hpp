#pragma once
#include <MetaObject/params/IParam.hpp>
#include "SerializationFactory.hpp"
#include <MetaObject/params/ITAccessibleParam.hpp>

#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/archives/json.hpp>
#include <functional>

namespace mo {
template<class T, class Enable = void> class ITParam;
template<class T, int N, typename Enable> struct MetaParam;
namespace IO {
namespace Cereal {
template<class T> struct Policy {
    Policy() {
        SerializationFactory::instance()->setBinarySerializationFunctions(
            TypeInfo(typeid(T)),
            std::bind(&Policy<T>::serialize<cereal::BinaryOutputArchive>, std::placeholders::_1, std::placeholders::_2),
            std::bind(&Policy<T>::deSerialize<cereal::BinaryInputArchive>, std::placeholders::_1, std::placeholders::_2));

        SerializationFactory::instance()->setXmlSerializationFunctions(
            TypeInfo(typeid(T)),
            std::bind(&Policy<T>::serialize<cereal::XMLOutputArchive>, std::placeholders::_1, std::placeholders::_2),
            std::bind(&Policy<T>::deSerialize<cereal::XMLInputArchive>, std::placeholders::_1, std::placeholders::_2));

        SerializationFactory::instance()->setJsonSerializationFunctions(
            TypeInfo(typeid(T)),
            std::bind(&Policy<T>::serialize<cereal::JSONOutputArchive>, std::placeholders::_1, std::placeholders::_2),
            std::bind(&Policy<T>::deSerialize<cereal::JSONInputArchive>, std::placeholders::_1, std::placeholders::_2));
    }

    template<class AR>
    static bool serialize(IParam* param, AR& ar) {
        auto typed = dynamic_cast<ITAccessibleParam<T>*>(param);
        if(typed == nullptr)
            return false;
        auto token = typed->access();
        ar(cereal::make_nvp(param->getName(), (token)()));
        return true;
    }
    template<class AR>
    static bool deSerialize(IParam* param, AR& ar) {
        auto typed = dynamic_cast<ITAccessibleParam<T>*>(param);
        if (typed == nullptr)
            return false;
        auto token = typed->access();
        auto nvp = cereal::make_optional_nvp(param->getName(), (token)(), (token)());
        try {
            ar(nvp);
        } catch(...) {
            token.setValid(false);
            return false;
        }

        if(nvp.success) {
            return true;
        }
        token.setValid(false);
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


