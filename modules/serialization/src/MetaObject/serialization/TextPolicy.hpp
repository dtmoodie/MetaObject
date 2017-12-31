#pragma once
#include "SerializationFactory.hpp"
#include <MetaObject/core/detail/Counter.hpp>
#include <MetaObject/core/detail/Forward.hpp>
#include <MetaObject/params/IParam.hpp>
#include <MetaObject/params/MetaParam.hpp>
#include <MetaObject/params/detail/MetaParamImpl.hpp>
#include <MetaObject/params/reflect_data.hpp>
#include <boost/lexical_cast.hpp>
#include <iomanip>
#include <map>

#define ASSERT_SERIALIZABLE(TYPE)                                                                                      \
    static_assert(mo::IO::Text::imp::stream_serializable<TYPE>::value, "Checking stream serializable for " #TYPE)

namespace mo
{
    namespace IO
    {
        namespace Text
        {
            namespace imp
            {
                inline size_t textSize(const std::string& str);
                inline size_t textSize(int value);

                template <class T>
                static inline mo::reflect::enable_if_reflected<T, size_t> textSizeHelper(const T& obj, mo::_counter_<0>)
                {
                    return textSize(mo::reflect::get<0>(obj));
                }

                template <class T, int I>
                static inline mo::reflect::enable_if_reflected<T, size_t> textSizeHelper(const T& obj, mo::_counter_<I>)
                {
                    return textSizeHelper(obj, mo::_counter_<I - 1>()) + textSize(mo::reflect::get<I>(obj));
                }

                template <class T>
                static inline mo::reflect::enable_if_reflected<T, size_t> textSize(const T& obj)
                {
                    return textSizeHelper(obj, mo::_counter_<mo::reflect::ReflectData<T>::N - 1>());
                }

                // test if stream serialization of a type is possible
                template <class T>
                struct stream_deserializable
                {
                    template <class U>
                    static constexpr auto check(std::istream* is, U* val) -> decltype(*is >> *val, size_t())
                    {
                        return 0;
                    }
                    template <class U>
                    static constexpr int check(...)
                    {
                        return 0;
                    }
                    static const bool value = sizeof(check<T>(static_cast<std::istream*>(nullptr),
                                                              static_cast<T*>(nullptr))) == sizeof(size_t);
                };

                template <class T>
                struct stream_serializable
                {
                    template <class U>
                    static constexpr auto check(std::ostream* os, U* val) -> decltype(*os << *val, size_t(0))
                    {
                        return 0;
                    }
                    template <class U>
                    static constexpr int check(...)
                    {
                        return 0;
                    }
                    static const bool value = sizeof(check<T>(static_cast<std::ostream*>(nullptr),
                                                              static_cast<T*>(nullptr))) == sizeof(size_t);
                };

                template <class T1, class T2>
                struct stream_serializable<std::pair<T1, T2>>
                {
                    static const bool value = stream_serializable<T1>::value && stream_serializable<T2>::value;
                };
            } // mo::IO::Text::imp
        }     // mo::IO::Text

        template <class T, class Enable = void>
        struct Traits
        {
        };

        template <class T, class Enable = void>
        struct PODTraits
        {
        };

        template <class T>
        struct PODTraits<T,
                         typename std::enable_if<Text::imp::stream_serializable<T>::value &&
                                                 Text::imp::stream_deserializable<T>::value>::type>
        {
            enum
            {
                // used in containers to determine the number of elements that can be displayed per line
                ElemsPerLine = 4,
            };

            static inline size_t textSize(const T& obj) { return Text::imp::textSize(obj); }

            static inline bool serialize(std::ostream& os, const T& obj)
            {
                os << obj;
                return true;
            }

            static inline bool deserialize(std::istream& is, T& obj)
            {
                is >> obj;
                return true;
            }
        };

        template <class T>
        struct PODTraits<T,
                         typename std::enable_if<Text::imp::stream_serializable<T>::value &&
                                                 !Text::imp::stream_deserializable<T>::value>::type>
        {
            enum
            {
                // used in containers to determine the number of elements that can be displayed per line
                ElemsPerLine = 4,
            };

            static inline size_t textSize(const T& obj) { return Text::imp::textSize(obj); }
            static inline bool serialize(std::ostream& os, const T& obj)
            {
                os << obj;
                return true;
            }
            static inline bool deserialize(std::istream& is, T& obj)
            {
                (void)is;
                (void)obj;
                return false;
            }
        };

        template <class T>
        struct PODTraits<T,
                         typename std::enable_if<!Text::imp::stream_serializable<T>::value &&
                                                 Text::imp::stream_deserializable<T>::value>::type>
        {
            enum
            {
                // used in containers to determine the number of elements that can be displayed per line
                ElemsPerLine = 4,
            };
            static inline size_t textSize(const T& obj) { return Text::imp::textSize(obj); }
            static inline bool serialize(std::ostream& os, const T& obj)
            {
                (void)os;
                (void)obj;
                return false;
            }
            static inline bool deserialize(std::istream& is, T& obj)
            {
                is >> obj;
                return true;
            }
        };

        template <class T>
        struct PODTraits<T,
                         typename std::enable_if<!Text::imp::stream_serializable<T>::value &&
                                                 !Text::imp::stream_deserializable<T>::value>::type>
        {
            enum
            {
                // used in containers to determine the number of elements that can be displayed per line
                ElemsPerLine = 4,
            };
            static inline size_t textSize(const T& obj)
            {
                (void)obj;
                return 0;
            }
            static inline bool serialize(std::ostream& os, const T& obj)
            {
                (void)os;
                (void)obj;
                return false;
            }
            static inline bool deserialize(std::istream& is, T& obj)
            {
                (void)is;
                (void)obj;
                return false;
            }
        };

        template <class T, int N, class Enable = void>
        struct TraitSelector
        {
            typedef typename TraitSelector<T, N - 1, void>::Trait Trait;
        };

        template <class T>
        struct TraitSelector<T, 0, void>
        {
            typedef PODTraits<T> Trait;
        };

        template <class T>
        struct ContainerTrait
        {
            enum
            {
                Default = 1
            };
        };

        template <class T>
        struct ContainerTrait<std::vector<T>>
        {
            enum
            {
                // used in containers to determine the number of elements that can be displayed per line
                ElemsPerLine = 4,
                Default = 0
            };
            static inline size_t textSize(const std::vector<T>& obj)
            {
                size_t output = 0;
                for (size_t i = 0; i < obj.size(); ++i)
                {
                    output += textSize(obj[i]);
                }
                return output;
            }
            static inline bool serialize(std::ostream& os, const std::vector<T>& obj)
            {
                os << "size = " << obj.size();
                os << '\n';

                size_t max_size = 0;
                for (const auto& item : obj)
                {
                    max_size = std::max<size_t>(max_size, mo::IO::TraitSelector<T, 3, void>::Trait::textSize(item));
                }
                max_size += 4;
                // index is 3 digits plus '=' sign
                int i = 0;
                int max_cols = 300 / (max_size + 5);
                if (max_cols == 0)
                    max_cols = 1;
                max_cols = std::max<int>(max_cols, mo::IO::TraitSelector<T, 3, void>::Trait::ElemsPerLine);

                while (i < obj.size()) // row
                {
                    int col_count = 0;
                    while (i < obj.size() && col_count < max_cols) // col
                    {
                        os << std::setw(3) << std::setfill('0') << i;
                        os << '=';
                        int size = mo::IO::TraitSelector<T, 3, void>::Trait::textSize(obj[i]);
                        for (int j = size; j < max_size; ++j)
                            os << ' ';
                        mo::IO::TraitSelector<T, 3, void>::Trait::serialize(os, obj[i]);
                        os << ' ';
                        ++col_count;
                        ++i;
                    }
                    os << '\n';
                }
                return true;
            }
            static inline bool deserialize(std::istream& is, std::vector<T>& obj)
            {
                std::string str;
                is >> str;
                auto pos = str.find('=');
                if (pos != std::string::npos)
                {
                    size_t index = boost::lexical_cast<size_t>(str.substr(0, pos));

                    std::stringstream ss;
                    ss << str.substr(pos + 1);
                    T value;
                    if (mo::IO::TraitSelector<T, 3, void>::Trait::deserialize(ss, value))
                    {
                        if (index < obj.size())
                        {
                            obj[index] = value;
                        }
                        else
                        {
                            obj.resize(index + 1);
                            obj[index] = value;
                        }
                        return true;
                    }
                }
                return false;
            }
        };
        template <class T1, class T2>
        struct ContainerTrait<std::pair<T1, T2>>
        {
            enum
            {
                // used in containers to determine the number of elements that can be displayed per line
                ElemsPerLine = 4,
                Default = 0
            };
            static inline size_t textSize(const std::pair<T1, T2>& obj)
            {
                size_t output = 0;
                for (size_t i = 0; i < obj.size(); ++i)
                {
                    output += textSize(obj[i]);
                }
                return output;
            }
            static inline bool serialize(std::ostream& os, const std::pair<T1, T2>& obj)
            {
                if (mo::IO::TraitSelector<T1, 3, void>::Trait::serialize(os, obj.first))
                {
                    os << ',';
                    if (mo::IO::TraitSelector<T2, 3, void>::Trait::serialize(os, obj.second))
                    {
                        return true;
                    }
                }
                return false;
            }
            static inline bool deserialize(std::istream& is, std::pair<T1, T2>& obj)
            {
                if (mo::IO::TraitSelector<T1, 3, void>::Trait::deserialize(is, obj.first))
                {
                    char c;
                    is >> c;
                    if (mo::IO::TraitSelector<T2, 3, void>::Trait::deserialize(is, obj.second))
                    {
                        return true;
                    }
                }
                return false;
            }
        };
        template <class K, class T>
        struct ContainerTrait<std::map<K, T>>
        {
            enum
            {
                // used in containers to determine the number of elements that can be displayed per line
                ElemsPerLine = 4,
                Default = 0
            };
            static inline size_t textSize(const std::map<K, T>& obj)
            {
                size_t output = 0;
                for (size_t i = 0; i < obj.size(); ++i)
                {
                    output += textSize(obj[i]);
                }
                return output;
            }
            static inline bool serialize(std::ostream& os, const std::map<K, T>& obj)
            {
                int count = 0;
                for (const auto& pair : obj)
                {
                    if (count != 0)
                        os << ", ";
                    mo::IO::TraitSelector<K, 3, void>::Trait::serialize(os, pair.first);
                    os << '=';
                    mo::IO::TraitSelector<T, 3, void>::Trait::serialize(os, pair.second);
                    ++count;
                }
                return true;
            }
            static inline bool deserialize(std::istream& is, std::map<K, T>& obj)
            {
                std::string str;
                is >> str;
                auto pos = str.find('=');
                if (pos == std::string::npos)
                    return false;
                K key;
                T value;
                {
                    std::stringstream ss;
                    ss << str.substr(0, pos);
                    mo::IO::TraitSelector<K, 3, void>::Trait::deserialize(ss, key);
                }
                {
                    std::stringstream ss;
                    ss << str.substr(pos + 1);
                    mo::IO::TraitSelector<T, 3, void>::Trait::deserialize(ss, value);
                }
                obj[key] = value;
                return true;
            }
        };

        template <class T>
        struct TraitSelector<T, 1, typename std::enable_if<ContainerTrait<T>::Default == 0>::type>
        {
            typedef ContainerTrait<T> Trait;
        };
        namespace Text
        {
            namespace imp
            {
                inline size_t textSize(const std::string& str) { return str.size(); }
                inline size_t textSize(int value)
                {
                    size_t sign = 0;
                    if (value < 0)
                        sign = 1;
                    value = abs(value);
                    if (value > 1000)
                        return 4 + sign;
                    if (value > 100)
                        return 3 + sign;
                    if (value > 10)
                        return 2 + sign;
                    return 1 + sign;
                }

                template <typename T>
                bool Serialize(ITAccessibleParam<T>* param, std::stringstream& ss)
                {
                    bool success = false;
                    try
                    {
                        auto token = param->access();
                        success = TraitSelector<T, 3>::Trait::serialize(ss, token());
                        token.setValid(false); // read only access
                    }
                    catch (...)
                    {
                    }

                    return success;
                }

                template <typename T>
                bool DeSerialize(ITAccessibleParam<T>* param, std::stringstream& ss)
                {
                    auto token = param->access();
                    return TraitSelector<T, 3>::Trait::deserialize(ss, token());
                }
            } // namespace imp

            template <typename T>
            bool WrapSerialize(IParam* param, std::stringstream& ss)
            {
                auto typed = dynamic_cast<ITAccessibleParam<T>*>(param);
                if (typed)
                {
                    if (imp::Serialize(typed, ss))
                    {
                        return true;
                    }
                }
                return false;
            }

            template <typename T>
            bool WrapDeSerialize(IParam* param, std::stringstream& ss)
            {
                auto typed = dynamic_cast<ITAccessibleParam<T>*>(param);
                if (typed)
                {
                    if (imp::DeSerialize(typed, ss))
                    {
                        typed->emitUpdate();
                        return true;
                    }
                }
                return false;
            }

            template <class T>
            struct Policy
            {
                Policy()
                {
                    SerializationFactory::instance()->setTextSerializationFunctions(
                        TypeInfo(typeid(T)),
                        std::bind(&WrapSerialize<T>, std::placeholders::_1, std::placeholders::_2),
                        std::bind(&WrapDeSerialize<T>, std::placeholders::_1, std::placeholders::_2));
                }
            };
        } // Text
    }     // IO

    /*#define PARAMETER_TEXT_SERIALIZATION_POLICY_INST_(N) \
        template <class T> \
        struct MetaParam<T, N, void> : public MetaParam<T, N - 1, void> \
        { \
            static IO::Text::Policy<T> _text_policy; \
            MetaParam(const char* name) : MetaParam<T, N - 1, void>(name) { (void)&_text_policy; } \
        }; \
        template <class T> \
        IO::Text::Policy<T> MetaParam<T, N, void>::_text_policy;

        PARAMETER_TEXT_SERIALIZATION_POLICY_INST_(__COUNTER__)
    */
} // mo
