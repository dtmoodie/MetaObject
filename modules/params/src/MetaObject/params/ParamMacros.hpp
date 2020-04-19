#pragma once

#include <MetaObject/core/detail/Enums.hpp>
#include <MetaObject/core/detail/HelperMacros.hpp>
#include <MetaObject/params/IParam.hpp>
#include <MetaObject/params/TParamPtr.hpp>
#include <MetaObject/params/TPublisher.hpp>
#include <MetaObject/params/TSubscriberPtr.hpp>
#include <MetaObject/params/detail/ParamMacrosImpl.hpp>

#define REFLECT_INTERNAL_WITH_FLAG(FLAGS, TYPE, NAME, INIT)                                                            \
    TYPE NAME = INIT;                                                                                                  \
    static inline TYPE initializer_##NAME()                                                                            \
    {                                                                                                                  \
        return INIT;                                                                                                   \
    }                                                                                                                  \
    constexpr static auto getPtr(const ct::Indexer<__COUNTER__ - REFLECT_COUNT_BEGIN>)                                 \
    {                                                                                                                  \
        return ct::makeMemberObjectPointer<FLAGS>(                                                                     \
            #NAME, &DataType::NAME, ct::makeInitializer(&DataType::initializer_##NAME, #INIT));                        \
    }

#define PARAM(TYPE, NAME, ...)                                                                                         \
    REFLECT_INTERNAL_MEMBER(mo::TControlParam<TYPE*>, NAME##_param)                                                    \
    REFLECT_INTERNAL_WITH_FLAG(mo::ParamReflectionFlags::kCONTROL, TYPE, NAME, __VA_ARGS__)

#define ENUM_PARAM(name, ...)                                                                                          \
    mo::TParamPtr<mo::EnumParam> name##_param;                                                                         \
    mo::EnumParam name;                                                                                                \
    ENUM_PARAM_(__COUNTER__, name, __VA_ARGS__)

#define RANGED_PARAM(type, name, init, min, max)

#define INPUT(TYPE, NAME)                                                                                              \
    REFLECT_INTERNAL_WITH_FLAG(mo::ParamReflectionFlags::kOUTPUT, mo::TSubscriberPtr<TYPE>::Input_t, NAME, nullptr)    \
    REFLECT_INTERNAL_MEMBER(mo::TSubscriberPtr<TYPE>, NAME##_param)

#define OPTIONAL_INPUT(type, name)                                                                                     \
    REFLECT_INTERNAL_WITH_FLAGS(mo::ParamReflectionFlags::kOUTPUT | mo::ParamReflectionFlags::kOPTIONAL,               \
                                mo::TSubscriberPtr<type_>::Input_t,                                                    \
                                nullptr)                                                                               \
    REFLECT_INTERNAL(mo::TSubscriberPtr<TYPE>, name##_param)

#define STATE(TYPE, NAME, ...)                                                                                         \
    REFLECT_INTERNAL_MEMBER(mo::TParamPtr<TYPE>, NAME##_param)                                                         \
    REFLECT_INTERNAL_WITH_FLAG(mo::ParamReflectionFlags::kSTATE, TYPE, NAME, __VA_ARGS__)

#define STATUS(TYPE, NAME, ...)                                                                                        \
    REFLECT_INTERNAL_MEMBER(mo::TParamPtr<TYPE>, NAME##_param)                                                         \
    REFLECT_INTERNAL_WITH_FLAG(mo::ParamReflectionFlags::kSTATUS, TYPE, NAME, __VA_ARGS__)

// These don't use CT's REFLECT_INTERNAL... macros because we want init##NAME to return TYPE whereas
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

#define SOURCE_1(TYPE, NAME)                                                                                           \
    mo::TPublisher<TYPE, mo::ParamFlags::kOUTPUT | mo::ParamFlags::kSOURCE> NAME;                                      \
    constexpr static auto getPtr(const ct::Indexer<__COUNTER__ - REFLECT_COUNT_BEGIN>)                                 \
    {                                                                                                                  \
        return ct::makeMemberObjectPointer<mo::ParamReflectionFlags::kSOURCE>(#NAME, &DataType::NAME);                 \
    }

#define SOURCE_2(TYPE, NAME, INIT)                                                                                     \
    mo::TFPublisher<TYPE, mo::ParamFlags::kOUTPUT | mo::ParamFlags::kSOURCE> NAME;                                     \
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
#define SOURCE(TYPE, ...) CT_PP_CAT(CT_PP_OVERLOAD(SOURCE_, __VA_ARGS__)(TYPE, __VA_ARGS__), CT_PP_EMPTY())
#else
#define SOURCE(TYPE, ...) CT_PP_OVERLOAD(SOURCE_, __VA_ARGS__)(TYPE, __VA_ARGS__)
#endif