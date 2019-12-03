#pragma once
#include <ct/VariadicTypedef.hpp>
#include <type_traits>
#include <cstdint>

#if defined(_WIN32) || defined(_WIN64)
#if _WIN64
#define ENVIRONMENT64
#else
#define ENVIRONMENT32
#endif
#endif

// Check GCC
#if __GNUC__
#if __x86_64__ || __ppc64__
#define ENVIRONMENT64
#else
#define ENVIRONMENT32
#endif
#endif

namespace mo
{
    using PrimitiveTypes = ct::VariadicTypedef<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t,
#ifdef ENVIRONMENT64
                                               long long, unsigned long long,
#else
                                               long int, unsigned long int,
#endif
                                               float, double, void*, char, bool>;

    template <class T>
    struct IsPrimitive
    {
        using type = typename std::remove_cv<T>::type;
        static constexpr const bool value = PrimitiveTypes::template contains<T>();
    };

}

#if __GNUG__ && __GNUC__ < 5



namespace std
{

    template<class T>
    struct is_trivially_copyable
    {
        static constexpr const bool value = __has_trivial_copy(T);
    };
}

#endif
