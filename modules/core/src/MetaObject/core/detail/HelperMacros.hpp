#ifndef MO_CORE_HELPER_MACROS_HPP
#define MO_CORE_HELPER_MACROS_HPP
#include <ct/macros.hpp>
#include <ct/type_traits.hpp>

#include <cstdint>


#define COMBINE1(X, Y) X##Y // helper macro
#define COMBINE(X, Y) COMBINE1(X, Y)

#define MO_STRINGIFY_1(X1) #X1
#define MO_STRINGIFY_2(X1, X2) #X1, #X2
#define MO_STRINGIFY_3(X1, X2, X3) #X1, #X2, #X3
#define MO_STRINGIFY_4(X1, X2, X3, X4) #X1, #X2, #X3, #X4
#define MO_STRINGIFY_5(X1, X2, X3, X4, X5) #X1, #X2, #X3, #X4, #X5
#define MO_STRINGIFY_6(X1, X2, X3, X4, X5, X6) #X1, #X2, #X3, #X4, #X5, #X6
#define MO_STRINGIFY_7(X1, X2, X3, X4, X5, X6, X7) #X1, #X2, #X3, #X4, #X5, #X6, #X7
#define MO_STRINGIFY_8(X1, X2, X3, X4, X5, X6, X7, X8) #X1, #X2, #X3, #X4, #X5, #X6, #X7, #X8
#define MO_STRINGIFY_9(X1, X2, X3, X4, X5, X6, X7, X8, X9) #X1, #X2, #X3, #X4, #X5, #X6, #X7, #X8, #X9

#ifdef _MSC_VER
#define MO_STRINGIFY(...) CT_PP_CAT(CT_PP_OVERLOAD(MO_STRINGIFY_, __VA_ARGS__)(__VA_ARGS__), BOOST_PP_EMPTY())
#else
#define MO_STRINGIFY(...) CT_PP_OVERLOAD(MO_STRINGIFY_, __VA_ARGS__)(__VA_ARGS__)
#endif

#define ENUM_EXPAND(...)                                                                                               \
    {MO_STRINGIFY(__VA_ARGS__)},                                                                                       \
    {                                                                                                                  \
        __VA_ARGS__                                                                                                    \
    }

#define DEFINE_MEMBER_DETECTOR(X)                                                                                      \
    template <typename T>                                                                                              \
    class Detect_##X                                                                                                   \
    {                                                                                                                  \
        struct Fallback                                                                                                \
        {                                                                                                              \
            int X;                                                                                                     \
        };                                                                                                             \
        struct Derived : T, Fallback                                                                                   \
        {                                                                                                              \
        };                                                                                                             \
        template <typename U, U>                                                                                       \
        struct Check;                                                                                                  \
        typedef char ArrayOfOne[1];                                                                                    \
        typedef char ArrayOfTwo[2];                                                                                    \
        template <typename U>                                                                                          \
        static ArrayOfOne& func(Check<int Fallback::*, &U::X>*);                                                       \
        template <typename U>                                                                                          \
        static ArrayOfTwo& func(...);                                                                                  \
                                                                                                                       \
      public:                                                                                                          \
        typedef Detect_##X type;                                                                                       \
        enum                                                                                                           \
        {                                                                                                              \
            value = sizeof(func<Derived>(0)) == 2                                                                      \
        };                                                                                                             \
    };

namespace mo
{
    template <class T>
    struct Void
    {
        using type = void;
    };
}

#define DEFINE_TYPEDEF_DETECTOR(typedef_NAME)                                                                          \
    template <class T, class U = void>                                                                                 \
    struct has_##typedef_NAME                                                                                          \
    {                                                                                                                  \
        enum                                                                                                           \
        {                                                                                                              \
            value = 0                                                                                                  \
        };                                                                                                             \
    };                                                                                                                 \
    template <class T>                                                                                                 \
    struct has_##typedef_NAME<T, typename mo::Void<typename T::PARENT_CLASS>::type>                                    \
    {                                                                                                                  \
        enum                                                                                                           \
        {                                                                                                              \
            value = 1                                                                                                  \
        };                                                                                                             \
    };

#endif
