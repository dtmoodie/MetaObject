#pragma once

#include <MetaObject/core/detail/Enums.hpp>
#include <MetaObject/core/detail/HelperMacros.hpp>

#include <MetaObject/params/IParam.hpp>
#include <MetaObject/params/TPublisher.hpp>
#include <MetaObject/params/TSubscriberPtr.hpp>
#include <MetaObject/params/detail/ParamMacrosImpl.hpp>

#include <MetaObject/types/file_types.hpp>

#define REFLECT_INTERNAL_WITH_FLAG(FLAGS, TYPE, NAME, ...)                                                             \
    TYPE NAME = __VA_ARGS__;                                                                                           \
    static inline TYPE initializer_##NAME()                                                                            \
    {                                                                                                                  \
        return __VA_ARGS__;                                                                                            \
    }                                                                                                                  \
    constexpr static auto getPtr(const ct::Indexer<__COUNTER__ - REFLECT_COUNT_BEGIN>)                                 \
    {                                                                                                                  \
        return ct::makeMemberObjectPointer<FLAGS>(                                                                     \
            #NAME, &DataType::NAME, ct::makeInitializer(&DataType::initializer_##NAME, ""));                           \
    }

#define PARAM(TYPE, NAME, ...)                                                                                         \
    REFLECT_INTERNAL_MEMBER(mo::TControlParam<TYPE*>, NAME##_param)                                                    \
    REFLECT_INTERNAL_WITH_FLAG(mo::ParamReflectionFlags::kCONTROL, TYPE, NAME, __VA_ARGS__)

#define ENUM_PARAM(name, ...)                                                                                          \
    mo::TControlParam<mo::EnumParam*> name##_param;                                                                    \
    mo::EnumParam name = mo::EnumParam{ENUM_EXPAND(__VA_ARGS__)};                                                      \
    PUBLIC_ACCESS(name);

#define INPUT(TYPE, NAME)                                                                                              \
    REFLECT_INTERNAL_WITH_FLAG(mo::ParamReflectionFlags::kINPUT, mo::TSubscriberPtr<TYPE>::Input_t, NAME, nullptr)     \
    REFLECT_INTERNAL_MEMBER(mo::TSubscriberPtr<TYPE>, NAME##_param)

#define FLAGGED_INPUT(FLAG, TYPE, NAME)                                                                                \
    REFLECT_INTERNAL_WITH_FLAG(mo::ParamReflectionFlags::kINPUT, mo::TSubscriberPtr<TYPE>::Input_t, NAME, nullptr)     \
    using NAME##_Subscriber_t = mo::TFSubscriberPtr<TYPE, FLAG>;                                                       \
    REFLECT_INTERNAL_MEMBER(NAME##_Subscriber_t, NAME##_param)

#define OPTIONAL_INPUT(type, name)                                                                                     \
    REFLECT_INTERNAL_WITH_FLAG(mo::ParamReflectionFlags::kINPUT | mo::ParamReflectionFlags::kOPTIONAL,                 \
                               mo::TSubscriberPtr<type>::Input_t,                                                      \
                               name,                                                                                   \
                               nullptr)                                                                                \
    using name##_Subscriber_t = mo::TFSubscriberPtr<type, mo::ParamFlags::kOPTIONAL>;                                  \
    REFLECT_INTERNAL_MEMBER(name##_Subscriber_t, name##_param)

#define STATE(TYPE, NAME, ...)                                                                                         \
    REFLECT_INTERNAL_MEMBER(mo::TControlParam<TYPE*>, NAME##_param)                                                    \
    REFLECT_INTERNAL_WITH_FLAG(mo::ParamReflectionFlags::kSTATE, TYPE, NAME, __VA_ARGS__)

#define STATUS(TYPE, NAME, ...)                                                                                        \
    REFLECT_INTERNAL_MEMBER(mo::TControlParam<TYPE*>, NAME##_param)                                                    \
    REFLECT_INTERNAL_WITH_FLAG(mo::ParamReflectionFlags::kSTATUS, TYPE, NAME, __VA_ARGS__)

// These don't use CT's REFLECT_INTERNAL... macros because we want init##NAME to return
// TYPE whereas
#define OUTPUT_1(TYPE, NAME)                                                                                           \
    mo::TPublisher<TYPE> NAME;                                                                                         \
    constexpr static auto getPtr(const ct::Indexer<__COUNTER__ - REFLECT_COUNT_BEGIN>)                                 \
    {                                                                                                                  \
        return ct::makeMemberObjectPointer(#NAME, &DataType::NAME);                                                    \
    }

#define OUTPUT_2(TYPE, NAME, INIT)                                                                                     \
    mo::TPublisher<TYPE> NAME;                                                                                         \
    inline static TYPE init##NAME()                                                                                    \
    {                                                                                                                  \
        return INIT;                                                                                                   \
    }                                                                                                                  \
    constexpr static auto getPtr(const ct::Indexer<__COUNTER__ - REFLECT_COUNT_BEGIN>)                                 \
    {                                                                                                                  \
        return ct::makeMemberObjectPointer(                                                                            \
            #NAME, &DataType::NAME, ct::Initializer<TYPE>(&DataType::init##NAME, #INIT));                              \
    }

#ifdef _MSC_VER
#define OUTPUT(TYPE, ...) CT_PP_CAT(CT_PP_OVERLOAD(OUTPUT_, __VA_ARGS__)(TYPE, __VA_ARGS__), CT_PP_EMPTY())
#else
#define OUTPUT(TYPE, ...) CT_PP_OVERLOAD(OUTPUT_, __VA_ARGS__)(TYPE, __VA_ARGS__)
#endif

#define OUTPUT_WITH_FLAG_1(TYPE, FLAG, NAME)                                                                           \
    mo::TFPublisher<TYPE, mo::ParamFlags::kOUTPUT | FLAG> NAME;                                                        \
    constexpr static auto getPtr(const ct::Indexer<__COUNTER__ - REFLECT_COUNT_BEGIN>)                                 \
    {                                                                                                                  \
        return ct::makeMemberObjectPointer<mo::ParamReflectionFlags::kSOURCE>(#NAME, &DataType::NAME);                 \
    }

#define OUTPUT_WITH_FLAG_2(TYPE, FLAG, NAME, INIT)                                                                     \
    mo::TFPublisher<TYPE, mo::ParamFlags::kOUTPUT | FLAG> NAME;                                                        \
    inline static TYPE init##NAME()                                                                                    \
    {                                                                                                                  \
        return INIT;                                                                                                   \
    }                                                                                                                  \
    constexpr static auto getPtr(const ct::Indexer<__COUNTER__ - REFLECT_COUNT_BEGIN>)                                 \
    {                                                                                                                  \
        return ct::makeMemberObjectPointer<mo::ParamReflectionFlags::kSOURCE>(                                         \
            #NAME, &DataType::NAME, ct::Initializer<TYPE>(&DataType::init##NAME, #INIT));                              \
    }

#ifdef _MSC_VER
#define OUTPUT_WITH_FLAG(TYPE, FLAG, ...)                                                                              \
    CT_PP_CAT(CT_PP_OVERLOAD(OUTPUT_WITH_FLAG_, __VA_ARGS__)(TYPE, FLAG, __VA_ARGS__), CT_PP_EMPTY())
#else
#define OUTPUT_WITH_FLAG(TYPE, FLAG, ...) CT_PP_OVERLOAD(OUTPUT_WITH_FLAG_, __VA_ARGS__)(TYPE, FLAG, __VA_ARGS__)
#endif

#define SOURCE(TYPE, ...) OUTPUT_WITH_FLAG(TYPE, mo::ParamFlags::kSOURCE, __VA_ARGS__)
