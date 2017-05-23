#pragma once
#include <MetaObject/Parameters/IParameter.hpp>
#include <MetaObject/Parameters/MetaParameter.hpp>
#include <MetaObject/Detail/Counter.hpp>
#include "SerializationFunctionRegistry.hpp"
#include <boost/lexical_cast.hpp>
#include <map>
#include <iomanip>
namespace mo
{
    template<class T> class ITypedParameter;
namespace IO
{
namespace Text
{
    namespace imp
    {
        inline int textSize(const std::string& str)
        {
            return str.size();
        }
        inline int textSize(int value)
        {
            int sign = 0;
            if(value < 0)
                sign = 1;
            value = abs(value);
            if(value > 1000)
                return 4 + sign;
            if(value > 100)
                return 3 + sign;
            if(value > 10)
                return 2 + sign;
            return 1 + sign;
        }

        // test if stream serialization of a type is possible
        template<class T>
        struct stream_deserializable{
            template<class U>
            static constexpr auto check(std::stringstream is, U val, int)->decltype(is >> val, size_t()){
                return 0;
            }
            template<class U>
            static constexpr int check(std::stringstream is, U val, size_t){
                return 0;
            }
            static const bool value = sizeof(check<T>(std::stringstream(), std::declval<T>(), 0)) == sizeof(size_t);
        };

        template<class T>
        struct stream_serializable{
            template<class U>
            static constexpr auto check(std::stringstream is, U val, int)->decltype(is << val, size_t()){
                return 0;
            }
            template<class U>
            static constexpr int check(std::stringstream is, U val, size_t){
                return 0;
            }
            static const bool value = sizeof(check<T>(std::stringstream(), std::declval<T>(), 0)) == sizeof(size_t);
        };

        template<class T1, class T2>
        struct stream_serializable<std::pair<T1, T2>>{
            static const bool value = stream_serializable<T1>::value && stream_serializable<T2>::value;
        };

        template<class T1, class T2>
        struct stream_deserializable<std::pair<T1, T2>>{
            static const bool value = stream_deserializable<T1>::value && stream_deserializable<T2>::value;
        };

        template<typename T>
        bool Serialize_imp(std::ostream& os, T const& obj, mo::_counter_<0> dummy){
            (void)dummy;
            return false;
        }

        template<typename T, int P>
        bool Serialize_imp(std::ostream& os, T const& obj, mo::_counter_<P> dummy){
            return Serialize_imp(os, obj, --dummy);
        }

        template<typename T>
        bool DeSerialize_imp(std::istream& os, T const& obj, mo::_counter_<0> dummy){
            (void)dummy;
            return false;
        }

        template<typename T, int P>
        bool DeSerialize_imp(std::istream& os, T const& obj, mo::_counter_<P> dummy){
            return DeSerialize_imp(os, obj, --dummy);
        }

        template<typename T>
        auto Serialize_imp(std::ostream& os, T const& obj, mo::_counter_<10> dummy) ->decltype(os << obj, bool())
        {
            os << obj;
            return true;
        }

        template<typename T1, typename T2> typename std::enable_if<stream_serializable<std::pair<T1, T2>>::value, bool>::type Serialize_imp(std::ostream& is, const std::pair<T1, T2>& obj, mo::_counter_<10> dumm7){
            is << obj.first;
            is << ',';
            is << obj.second;
            return true;
        }
        template<typename T>
        auto DeSerialize_imp(std::istream& is, T& obj, mo::_counter_<10> dummy) ->decltype(is >> obj, bool())
        {
            is >> obj;
            return true;
        }
        template<typename T1, typename T2> bool DeSerialize_imp(std::istream& is, std::pair<T1, T2>& obj, typename std::enable_if<stream_serializable<std::pair<T1, T2>>::value, mo::_counter_<10> >::type){
            is >> obj.first;
            char c;
            is >> c;
            is >> obj.second;
            return true;
        }

        template<typename T>
        auto Serialize_imp(std::ostream& os, std::vector<T> const& obj, mo::_counter_<10> dummy)->decltype(os << std::declval<T>(), bool()){
            os << "size = " << obj.size();
            os << '\n';

            int max_size = 0;
            for(const auto& item : obj)
            {
                max_size = std::max(max_size, textSize(item));
            }
            max_size += 4;
            // index is 3 digits plus '=' sign
            int i = 0;
            while( i < obj.size()) // row
            {
                int col_count = 0;
                while( i < obj.size() && col_count < 6) // col
                {
                    os << std::setw(3) << std::setfill('0') << i;
                    os << '=';
                    int size = textSize(obj[i]);
                    for(int j = size; j < max_size; ++j)
                        os << ' ';
                    os << obj[i] << ' ';
                    ++col_count;
                    ++i;
                }
                os << '\n';
            }
            return true;
        }

        template<typename T>
        auto DeSerialize_imp(std::istream& is, std::vector<T>& obj, mo::_counter_<10> dummy) ->decltype(is >> std::declval<T>(), bool())
        {
            std::string str;
            is >> str;
            auto pos = str.find('=');
            if(pos != std::string::npos)
            {
                size_t index = boost::lexical_cast<size_t>(str.substr(0, pos));
                T value = boost::lexical_cast<T>(str.substr(pos + 1));
                if(index < obj.size())
                {

                }else
                {
                    obj.resize(index + 1);
                    obj[index] = value;
                }
                return true;
            }
            return false;
        }

        template<class T1, class T2>
        typename std::enable_if<stream_serializable<T1>::value && stream_serializable<T2>::value, bool >::type
        DeSerialize_imp(std::istream& is, std::map<T1, T2>& obj, mo::_counter_<10> dummy){
            /*auto start = is.tellg();
            auto idx = start;
            char c = is.get();
            bool found = false;
            while(is.good()){
                if(c == '='){
                    found = true;
                    break;
                }
                idx = is.tellg();
                c = is.get();
            }
            if(!found) return;
            is.seekg(start);
            std::stringstream ss;
            while(is.tellg() != idx)
                ss << is.get();
            T1 key;
            DeSerialize_imp(ss, key, 0);
            ss.str(std::string());
            is.get();
            while(is.good())
                ss << is.get();
            T2 value;
            DeSerialize_imp(ss, value, 0);
            obj[key] = value;*/

            std::string str;
            is >> str;
            auto pos = str.find('=');
            if(pos == std::string::npos)
                return false;
            T1 key;
            T2 value;
            {
                std::stringstream ss;
                ss << str.substr(0, pos);
                DeSerialize_imp(ss, key, mo::_counter_<10>());
            }
            {
                std::stringstream ss;
                ss << str.substr(pos + 1);
                DeSerialize_imp(ss, value, mo::_counter_<10>());
            }
            obj[key] = value;
            return true;
        }

        template<class T1, class T2>
        typename std::enable_if<stream_serializable<T1>::value && stream_serializable<T2>::value, bool >::type
        Serialize_imp(std::ostream& os, std::map<T1, T2> const& obj, mo::_counter_<10> dummy){
            int count = 0;
            for(const auto& pair : obj){
                if(count != 0)
                    os << ", ";
                //os << pair.first << "=" << pair.second;
                Serialize_imp(os, pair.first, mo::_counter_<10>());
                os << '=';
                Serialize_imp(os, pair.second, mo::_counter_<10>());
                ++count;
            }
            return true;
        }

        template<typename T> bool DeSerialize_imp(std::stringstream& ss, std::vector<T>& param, typename std::enable_if<stream_deserializable<T>::value, mo::_counter_<10> >::type dummy){
            auto pos = ss.str().find('=');
            if(pos != std::string::npos){
                std::string str;
                std::getline(ss, str, '=');
                size_t index = boost::lexical_cast<size_t>(str);
                std::getline(ss, str);
                T value = boost::lexical_cast<T>(str);
                if(index >= param.size()){
                    param.resize(index + 1);
                }
                param[index] = value;
                return true;
            }else{
                param.clear();
                std::string size;
                std::getline(ss, size, '[');
                if (size.size()){
                    param.reserve(boost::lexical_cast<size_t>(size));
                }
                T value;
                char ch; // For flushing the ','
                while (ss >> value){
                    ss >> ch;
                    param.push_back(value);
                }
            }
            return true;
        }

        template<typename T>
        bool Serialize(ITypedParameter<T>* param, std::stringstream& ss){
            T* ptr = param->GetDataPtr();
            if (ptr){
                return Serialize_imp(ss, *ptr, mo::_counter_<10>());
            }
            return false;
        }

        template<typename T>
        bool DeSerialize(ITypedParameter<T>* param, std::stringstream& ss){
            T* ptr = param->GetDataPtr();
            if (ptr){
                return DeSerialize_imp(ss, *ptr, mo::_counter_<10>());
            }
            return false;
        }
    } // namespace imp


    template<typename T> bool WrapSerialize(IParameter* param, std::stringstream& ss)
    {
        ITypedParameter<T>* typed = dynamic_cast<ITypedParameter<T>*>(param);
        if (typed)
        {
            if(imp::Serialize(typed, ss))
            {
                return true;
            }
        }
        return false;
    }

    template<typename T> bool WrapDeSerialize(IParameter* param, std::stringstream& ss)
    {
        ITypedParameter<T>* typed = dynamic_cast<ITypedParameter<T>*>(param);
        if (typed)
        {
            if(imp::DeSerialize(typed, ss))
            {
                typed->Commit();
                return true;
            }
        }
        return false;
    }

    template<class T> struct Policy
    {
        Policy()
        {
            SerializationFunctionRegistry::Instance()->SetTextSerializationFunctions(
                TypeInfo(typeid(T)),
                std::bind(&WrapSerialize<T>, std::placeholders::_1, std::placeholders::_2),
                std::bind(&WrapDeSerialize<T>, std::placeholders::_1, std::placeholders::_2));
        }
    };
} // Text
} // IO

#define PARAMETER_TEXT_SERIALIZATION_POLICY_INST_(N) \
  template<class T> struct MetaParameter<T, N, void>: public MetaParameter<T, N - 1, void> \
    { \
        static IO::Text::Policy<T> _text_policy;  \
        MetaParameter(const char* name): \
            MetaParameter<T, N-1, void>(name) \
        { \
            (void)&_text_policy; \
        } \
    }; \
    template<class T> IO::Text::Policy<T> MetaParameter<T, N, void>::_text_policy;

PARAMETER_TEXT_SERIALIZATION_POLICY_INST_(__COUNTER__)
} // mo
