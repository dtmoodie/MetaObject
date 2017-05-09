#pragma once
#include "MetaObject/Detail/HelperMacros.hpp"
#include "MetaObject/Params/Demangle.hpp"

namespace mo {
template<class T, int N, typename Enable = void> struct MetaParam: public MetaParam<T, N-1, void> {
    MetaParam(const char* name = nullptr):
        MetaParam<T, N-1>(name) {}
};
template<class T> struct MetaParam<T, 0, void> {
    MetaParam(const char* name = nullptr) {
        if(name)
            mo::Demangle::RegisterName(mo::TypeInfo(typeid(T)), name);
    }
};
}
