#pragma once
#include "ParamFactory.hpp"
#include "MetaObject/Detail/Enums.hpp"
namespace mo {
template<class T> class ParamConstructor {
public:
    ParamConstructor() {
        ParamFactory::instance()->RegisterConstructor(TypeInfo(typeid(typename T::ValueType)),
                std::bind(&ParamConstructor<T>::create), T::Type);

        ParamFactory::instance()->RegisterConstructor(TypeInfo(typeid(T)),
                std::bind(&ParamConstructor<T>::create));
    }
    static IParam* create() {
        return new T();
    }
};
}