#pragma once

#include "MetaObject/serialization.hpp"
#include "SerializationFactory.hpp"
#include <MetaObject/params/IParam.hpp>
#include <MetaObject/params/ITAccessibleParam.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/cereal.hpp>
#include <functional>

namespace mo
{
    template <class T, int N, typename Enable>
    struct MetaParam;
    namespace IO
    {
        namespace Cereal
        {
            template <class T>
            struct Policy
            {
                Policy()
                {
                    SerializationFactory::instance()->setBinarySerializationFunctions(
                        TypeInfo(typeid(T)),
                        std::bind(&Policy<T>::serialize<cereal::BinaryOutputArchive>,
                                  std::placeholders::_1,
                                  std::placeholders::_2),
                        std::bind(&Policy<T>::deSerialize<cereal::BinaryInputArchive>,
                                  std::placeholders::_1,
                                  std::placeholders::_2));

                    SerializationFactory::instance()->setXmlSerializationFunctions(
                        TypeInfo(typeid(T)),
                        std::bind(&Policy<T>::serialize<cereal::XMLOutputArchive>,
                                  std::placeholders::_1,
                                  std::placeholders::_2),
                        std::bind(&Policy<T>::deSerialize<cereal::XMLInputArchive>,
                                  std::placeholders::_1,
                                  std::placeholders::_2));

                    SerializationFactory::instance()->setJsonSerializationFunctions(
                        TypeInfo(typeid(T)),
                        std::bind(&Policy<T>::serialize<cereal::JSONOutputArchive>,
                                  std::placeholders::_1,
                                  std::placeholders::_2),
                        std::bind(&Policy<T>::deSerialize<cereal::JSONInputArchive>,
                                  std::placeholders::_1,
                                  std::placeholders::_2));
                }

                template <class AR>
                static bool serialize(const IParam* param, AR& ar)
                {
                    auto typed = dynamic_cast<const TParam<T>*>(param);
                    if (typed == nullptr)
                    {
                        return false;
                    }
                    auto token = typed->read();
                    ar(cereal::make_nvp(param->getName(), (token)()));
                    return true;
                }

                template <class AR>
                static bool deSerialize(IParam* param, AR& ar)
                {
                    auto typed = dynamic_cast<TParam<T>*>(param);
                    if (typed == nullptr)
                    {
                        return false;
                    }
                    auto token = typed->access();
                    auto nvp = cereal::make_optional_nvp(param->getName(), (token)(), (token)());
                    try
                    {
                        ar(nvp);
                    }
                    catch (cereal::Exception& e)
                    {
                        MO_LOG(debug, "Failed to deserialize {} due to {}",  param->getName(), e.what());
                        token.setModified(false);
                        return false;
                    }
                    catch (...)
                    {
                        token.setModified(false);
                        return false;
                    }

                    if (nvp.success)
                    {
                        return true;
                    }
                    token.setModified(false);
                    return false;
                }
            };
        } // namespace Cereal
    }     // namespace IO
    template <class T>
    using DetectSerializer = typename std::enable_if<
        cereal::traits::detail::count_input_serializers<T, cereal::JSONInputArchive>::value != 0 &&
        cereal::traits::detail::count_input_serializers<T, cereal::XMLInputArchive>::value != 0 &&
        cereal::traits::detail::count_input_serializers<T, cereal::BinaryInputArchive>::value != 0>::type;
// template<class T> using DetectSerializer = void;

#define PARAM_CEREAL_SERIALIZATION_POLICY_INST_(N)                                                                     \
    template <class T>                                                                                                 \
    struct MetaParam<T, N, DetectSerializer<T>> : public MetaParam<T, N - 1, void>                                     \
    {                                                                                                                  \
        static IO::Cereal::Policy<T> _cereal_policy;                                                                   \
        MetaParam(SystemTable* table, const char* name)                                                                \
            : MetaParam<T, N - 1, void>(table, name)                                                                   \
        {                                                                                                              \
            (void)&_cereal_policy;                                                                                     \
        }                                                                                                              \
    };                                                                                                                 \
    template <class T>                                                                                                 \
    IO::Cereal::Policy<T> MetaParam<T, N, DetectSerializer<T>>::_cereal_policy;

    PARAM_CEREAL_SERIALIZATION_POLICY_INST_(__COUNTER__)
}
