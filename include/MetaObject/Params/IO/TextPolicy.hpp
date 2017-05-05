#pragma once
#include <MetaObject/Params/IParam.hpp>
#include <MetaObject/Params/MetaParam.hpp>
#include "SerializationFactory.hpp"
#include <boost/lexical_cast.hpp>
#include <map>
#include <iomanip>
namespace mo {
template<class T> class ITParam;
namespace IO {
namespace Text {
namespace imp {
inline int textSize(const std::string& str) {
    return str.size();
}
inline int textSize(int value) {
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
void Serialize_imp(std::ostream& os, T const& obj, long) {

}

template<typename T>
auto DeSerialize_imp(std::istream& is, T& obj, int) ->decltype(is >> obj, void()) {
    is >> obj;
}
template<typename T>
void DeSerialize_imp(std::istream& is, T& obj, long) {

}

template<typename T>
auto Serialize_imp(std::ostream& os, std::vector<T> const& obj, int)->decltype(os << std::declval<T>(), void()) {
    os << "size = " << obj.size();
    os << '\n';

    int max_size = 0;
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
            int size = textSize(obj[i]);
            for(int j = size; j < max_size; ++j)
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
bool Serialize(ITParam<T>* param, std::stringstream& ss) {
    T* ptr = param->GetDataPtr();
    if (ptr) {
        Serialize_imp(ss, *ptr, 0);
        //ss << *ptr;
        return true;
    }
    return false;
}

template<typename T>
bool DeSerialize(ITParam<T>* param, std::stringstream& ss) {
    T* ptr = param->GetDataPtr();
    if (ptr) {
        //ss >> *ptr;
        DeSerialize_imp(ss, *ptr, 0);
        return true;
    }
    return false;
}
template<typename T> bool Serialize(ITParam<std::vector<T>>* param, std::stringstream& ss) {
    std::vector<T>* ptr = param->GetDataPtr();
    if (ptr) {
        Serialize_imp(ss, *ptr, 0);
        return true;
    }
    return false;
}
template<typename T> bool DeSerialize(ITParam<std::vector<T>>* param, std::stringstream& ss) {
    std::vector<T>* ptr = param->GetDataPtr();
    if (ptr) {
        auto pos = ss.str().find('=');
        if(pos != std::string::npos) {
            std::string str;
            std::getline(ss, str, '=');
            size_t index = boost::lexical_cast<size_t>(str);
            std::getline(ss, str);
            T value = boost::lexical_cast<T>(str);
            if(index >= ptr->size()) {
                ptr->resize(index + 1);
            }
            (*ptr)[index] = value;
            return true;
        } else {
            ptr->clear();
            std::string size;
            std::getline(ss, size, '[');
            if (size.size()) {
                ptr->reserve(boost::lexical_cast<size_t>(size));
            }
            T value;
            char ch; // For flushing the ','
            while (ss >> value) {
                ss >> ch;
                ptr->push_back(value);
            }
        }
        return true;
    }
    return false;
}
}


template<typename T> bool WrapSerialize(IParam* param, std::stringstream& ss) {
    ITParam<T>* typed = dynamic_cast<ITParam<T>*>(param);
    if (typed) {
        if(imp::Serialize(typed, ss)) {
            return true;
        }
    }
    return false;
}

template<typename T> bool WrapDeSerialize(IParam* param, std::stringstream& ss) {
    ITParam<T>* typed = dynamic_cast<ITParam<T>*>(param);
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
        SerializationFactory::Instance()->SetTextSerializationFunctions(
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
