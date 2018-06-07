#pragma once
#include <utility>

namespace mo
{
    template <typename T, typename Tag = T>
    class NamedType
    {
    public:
        explicit NamedType(T* value) : _value(value) {}
        NamedType(NamedType&& other): _value(other._value){}
        T* get() const { return _value; }
    private:
        T* _value;
    };
}
