#pragma once
#include <cstdint>

#define COMBINE1(X,Y) X##Y  // helper macro
#define COMBINE(X,Y) COMBINE1(X,Y)

#define DEFINE_HAS_STATIC_FUNCTION(traitsName, funcName, signature)            \
    template <typename U>                                                      \
    class traitsName                                                           \
    {                                                                          \
    private:                                                                   \
        template<typename T, T> struct helper;                                 \
        template<typename T>                                                   \
        static std::uint8_t check(helper<signature, &funcName>*);              \
        template<typename T> static std::uint16_t check(...);                  \
    public:                                                                    \
        static const bool value = sizeof(check<U>(0)) == sizeof(std::uint8_t); \
    }

#define DEFINE_MEMBER_DETECTOR(X)                                                   \
template<typename T> class Detect_##X {                                             \
    struct Fallback { int X; };                                                     \
    struct Derived : T, Fallback { };                                               \
                                                                                    \
    template<typename U, U> struct Check;                                           \
                                                                                    \
    typedef char ArrayOfOne[1];                                                     \
    typedef char ArrayOfTwo[2];                                                     \
                                                                                    \
    template<typename U> static ArrayOfOne & func(Check<int Fallback::*, &U::X> *); \
    template<typename U> static ArrayOfTwo & func(...);                             \
  public:                                                                           \
    typedef Detect_##X type;                                                        \
    enum { value = sizeof(func<Derived>(0)) == 2 };                                 \
};

namespace mo
{
    template<class T>
    struct Void 
    {
        typedef void type;
    };

#define DEFINE_TYPEDEF_DETECTOR(TYPEDEF_NAME)                                       \
    template<class T, class U = void>                                               \
    struct has_##TYPEDEF_NAME                                                       \
    {                                                                               \
        enum { value = 0 };                                                         \
    };                                                                              \
    template<class T>                                                               \
    struct has_##TYPEDEF_NAME<T, typename Void<typename T::PARENT_CLASS>::type >    \
    {                                                                               \
        enum { value = 1 };                                                         \
    };

}