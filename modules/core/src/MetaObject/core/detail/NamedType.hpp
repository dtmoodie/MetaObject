#pragma once
#include "MetaObject/detail/TypeInfo.hpp"
#include <utility>

namespace mo
{
    template <typename T, typename Tag = T>
    class NamedType
    {
      public:
        explicit NamedType(T* value) : _value(value) {}
        NamedType(NamedType&& other) : _value(other._value) {}
        T* get() const { return _value; }
      private:
        T* _value;
    };

    struct Function;
    struct StaticFunction;

    template <class T, class R, class... Args, class Tag>
    class NamedType<R (T::*)(Args...), Tag>
    {
      public:
        using FPtr = R (T::*)(Args...);

        explicit NamedType(FPtr value) : _value(value) {}
        NamedType(NamedType&& other) : _value(other._value) {}
        FPtr get() const { return _value; }
      private:
        FPtr _value;
    };

    template <class R, class... Args, class Tag>
    class NamedType<R (*)(Args...), Tag>
    {
      public:
        using FPtr = R (*)(Args...);

        explicit NamedType(FPtr value) : _value(value) {}
        NamedType(NamedType&& other) : _value(other._value) {}
        FPtr get() const { return _value; }
      private:
        FPtr _value;
    };

    template <>
    class NamedType<const TypeInfo>
    {
      public:
        explicit NamedType(const TypeInfo value) : _value(value) {}
        NamedType(NamedType&& other) : _value(other._value) {}
        TypeInfo get() const { return _value; }
      private:
        TypeInfo _value;
    };
}
