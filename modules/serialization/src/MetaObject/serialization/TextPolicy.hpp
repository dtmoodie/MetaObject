#pragma once
#include <MetaObject/params/ITAccessibleParam.hpp>
#include <MetaObject/params/MetaParam.hpp>
#include "SerializationFactory.hpp"
#include <boost/lexical_cast.hpp>
#include <map>
#include <iomanip>
#include <type_traits>

namespace mo {
template<class T> class ITParam;
namespace IO {
namespace Text {
namespace imp {
template<class T>
constexpr bool is_numeric_v = (std::is_floating_point<T>::value || std::is_integral<T>::value);

inline size_t textSize(const std::string& str) {
    return str.size();
}

template<class T>
size_t textSize(T value, typename std::enable_if<is_numeric_v<T> && std::is_signed<T>::value, void>::type* dummy = 0) {
    int sign = 0;
    if(value < 0)
        sign = 1;
    value = std::abs<T>(value);
    if(value > 1000)
        return 4 + sign;
    if(value > 100)
        return 3 + sign;
    if(value > 10)
        return 2 + sign;
    return 1 + sign;
}

template<class T>
size_t textSize(T value, typename std::enable_if<is_numeric_v<T> && !std::is_signed<T>::value, void>::type* dummy = 0) {
    int sign = 0;
    if (value < 0)
        sign = 1;
    if (value > 1000)
        return 4 + sign;
    if (value > 100)
        return 3 + sign;
    if (value > 10)
        return 2 + sign;
    return 1 + sign;
}

// test if stream serialization of a type is possible
template<class T>
struct stream_serializable {
    template<class U>
    static constexpr auto check(std::stringstream is, U val, int)->decltype(is >> val, size_t()) {
        return 0;
    }
    template<class U>
    static constexpr int check(std::stringstream is, U val, size_t) {
        return 0;
    }
    static const bool value = sizeof(check<T>(std::stringstream(), std::declval<T>(), 0)) == sizeof(size_t);
};


template<typename T>
auto Serialize_imp(std::ostream& os, T const& obj, int) ->decltype(os << obj, void()) {
    os << obj;
}

template<typename T>
void Serialize_imp(std::ostream& os, T const& obj, size_t) {

}

template<typename T>
auto DeSerialize_imp(std::istream& is, T& obj, int) ->decltype(is >> obj, void()) {
    is >> obj;
}
template<typename T>
void DeSerialize_imp(std::istream& is, T& obj,size_t) {

}

template<typename T>
auto Serialize_imp(std::ostream& os, std::vector<T> const& obj, int)->decltype(os << std::declval<T>(), void()) {
    os << "size = " << obj.size();
    os << '\n';

    size_t max_size = 0;
    for(const auto& item : obj) {
        max_size = std::max(max_size, textSize(item));
    }
    max_size += 4;
    // index is 3 digits plus '=' sign
    int i = 0;
    while( i < obj.size()) { // row
        int col_count = 0;
        while( i < obj.size() && col_count < 6) { // col
            os << std::setw(3) << std::setfill('0') << i;
            os << '=';
            size_t size = textSize(obj[i]);
            for(size_t j = size; j < max_size; ++j)
                os << ' ';
            os << obj[i] << ' ';
            ++col_count;
            ++i;
        }
        os << '\n';
    }
}

template<typename T>
auto DeSerialize_imp(std::istream& is, std::vector<T>& obj, int) ->decltype(is >> std::declval<T>(), void()) {
    std::string str;
    is >> str;
    auto pos = str.find('=');
    if(pos != std::string::npos) {
        size_t index = boost::lexical_cast<size_t>(str.substr(0, pos));
        T value = boost::lexical_cast<T>(str.substr(pos + 1));
        if(index < obj.size()) {

        } else {
            obj.resize(index + 1);
            obj[index] = value;
        }
    }
}

template<class T1, class T2>
typename std::enable_if<stream_serializable<T1>::value && stream_serializable<T2>::value >::type
DeSerialize_imp(std::istream& is, std::map<T1, T2>& obj, int) {
    std::string str;
    is >> str;
    auto pos = str.find('=');
    if(pos == std::string::npos)
        return;
    T1 key;
    T2 value;
    {
        std::stringstream ss;
        ss << str.substr(0, pos);
        ss >> key;
    }
    {
        std::stringstream ss;
        ss << str.substr(pos + 1);
        ss >> value;
    }
    obj[key] = value;
}

template<class T1, class T2>
typename std::enable_if<stream_serializable<T1>::value && stream_serializable<T2>::value >::type
Serialize_imp(std::ostream& os, std::map<T1, T2> const& obj, int) {
    int count = 0;
    for(const auto& pair : obj) {
        if(count != 0)
            os << ", ";
        os << pair.first << "=" << pair.second;
        ++count;
    }
}


template<typename T>
bool Serialize(ITAccessibleParam<T>* param, std::stringstream& ss) {
    auto token = param->access();
    Serialize_imp(ss, (token)(), 0);
    return true;
}

template<typename T>
bool DeSerialize(ITAccessibleParam<T>* param, std::stringstream& ss) {
    auto token = param->access();
    DeSerialize_imp(ss, (token)(), 0);
    return true;
}
template<typename T> bool Serialize(ITAccessibleParam<std::vector<T>>* param, std::stringstream& ss) {
    auto token = param->access();
    Serialize_imp(ss, (token)(), 0);
    return true;
}
template<typename T> bool DeSerialize(ITAccessibleParam<std::vector<T>>* param, std::stringstream& ss) {
    auto token = param->access();
    auto pos = ss.str().find('=');
    if(pos != std::string::npos) {
        std::string str;
        std::getline(ss, str, '=');
        size_t index = boost::lexical_cast<size_t>(str);
        std::getline(ss, str);
        T value = boost::lexical_cast<T>(str);
        if(index >= (token)().size()) {
            (token)().resize(index + 1);
        }
        (token)()[index] = value;
        return true;
    } else {
        (token)().clear();
        std::string size;
        std::getline(ss, size, '[');
        if (size.size()) {
            (token)().reserve(boost::lexical_cast<size_t>(size));
        }
        T value;
        char ch; // For flushing the ','
        while (ss >> value) {
            ss >> ch;
            (token)().push_back(value);
        }
    }
    return true;
}
}


template<typename T> bool WrapSerialize(IParam* param, std::stringstream& ss) {
    auto typed = dynamic_cast<ITAccessibleParam<T>*>(param);
    if (typed) {
        if(imp::Serialize(typed, ss)) {
            return true;
        }
    }
    return false;
}

template<typename T> bool WrapDeSerialize(IParam* param, std::stringstream& ss) {
    auto typed = dynamic_cast<ITAccessibleParam<T>*>(param);
    if (typed) {
        if(imp::DeSerialize(typed, ss)) {
            typed->emitUpdate();
            return true;
        }
    }
    return false;
}

template<class T> struct Policy {
    Policy() {
        SerializationFactory::instance()->setTextSerializationFunctions(
            TypeInfo(typeid(T)),
            std::bind(&WrapSerialize<T>, std::placeholders::_1, std::placeholders::_2),
            std::bind(&WrapDeSerialize<T>, std::placeholders::_1, std::placeholders::_2));
    }
};
} // Text
} // IO

#define Param_TEXT_SERIALIZATION_POLICY_INST_(N) \
  template<class T> struct MetaParam<T, N, void>: public MetaParam<T, N - 1, void> \
    { \
        static IO::Text::Policy<T> _text_policy;  \
        MetaParam(const char* name): \
            MetaParam<T, N-1, void>(name) \
        { \
            (void)&_text_policy; \
        } \
    }; \
    template<class T> IO::Text::Policy<T> MetaParam<T, N, void>::_text_policy;

Param_TEXT_SERIALIZATION_POLICY_INST_(__COUNTER__)
} // mo
