#pragma once

#include <MetaObject/core/detail/Enums.hpp>
#include <MetaObject/core/detail/HelperMacros.hpp>
#include <MetaObject/params/IParam.hpp>
#include <MetaObject/params/TInputParam.hpp>
#include <MetaObject/params/TParamOutput.hpp>
#include <MetaObject/params/TParamPtr.hpp>
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
    REFLECT_INTERNAL_MEMBER(mo::TParamPtr<TYPE>, NAME##_param)                                                         \
    REFLECT_INTERNAL_WITH_FLAG(mo::ParamReflectionFlags::kCONTROL, TYPE, NAME, __VA_ARGS__)

#define ENUM_PARAM(name, ...)                                                                                          \
    mo::TParamPtr<mo::EnumParam> name##_param;                                                                         \
    mo::EnumParam name;                                                                                                \
    ENUM_PARAM_(__COUNTER__, name, __VA_ARGS__)

#define RANGED_PARAM(type, name, init, min, max)

#define INPUT(TYPE, NAME)                                                                                              \
    REFLECT_INTERNAL_WITH_FLAG(mo::ParamReflectionFlags::kOUTPUT, mo::TInputParamPtr<TYPE>::Input_t, NAME, nullptr)    \
    REFLECT_INTERNAL_MEMBER(mo::TInputParamPtr<TYPE>, NAME##_param)

#define OPTIONAL_INPUT(type, name)                                                                                     \
    REFLECT_INTERNAL_WITH_FLAGS(mo::ParamReflectionFlags::kOUTPUT | mo::ParamReflectionFlags::kOPTIONAL,               \
                                mo::TInputParamPtr<type_>::Input_t,                                                    \
                                nullptr)                                                                               \
    REFLECT_INTERNAL(mo::TInputParamPtr<TYPE>, name##_param)

#define STATE(TYPE, NAME, ...)                                                                                         \
    REFLECT_INTERNAL_MEMBER(mo::TParamPtr<TYPE>, NAME##_param)                                                         \
    REFLECT_INTERNAL_WITH_FLAG(mo::ParamReflectionFlags::kSTATE, TYPE, NAME, __VA_ARGS__)

#define STATUS(TYPE, NAME, ...)                                                                                        \
    REFLECT_INTERNAL_MEMBER(mo::TParamPtr<TYPE>, NAME##_param)                                                         \
    REFLECT_INTERNAL_WITH_FLAG(mo::ParamReflectionFlags::kSTATUS, TYPE, NAME, __VA_ARGS__)

// These don't use CT's REFLECT_INTERNAL... macros because we want init##NAME to return TYPE whereas
#define OUTPUT(TYPE, NAME, INIT)                                                                                       \
    mo::TParamOutput<TYPE> NAME;                                                                                       \
    inline static TYPE init##NAME()                                                                                    \
    {                                                                                                                  \
        return INIT;                                                                                                   \
    }                                                                                                                  \
    constexpr static auto getPtr(const ct::Indexer<__COUNTER__ - REFLECT_COUNT_BEGIN>)                                 \
    {                                                                                                                  \
        return ct::makeMemberObjectPointer(                                                                            \
            #NAME, &DataType::NAME, ct::Initializer<TYPE>(&DataType::init##NAME, #INIT));                              \
    }

#define SOURCE(TYPE, NAME, INIT)                                                                                       \
    mo::TParamOutput<TYPE, mo::ParamFlags::kOUTPUT | mo::ParamFlags::kSOURCE> NAME;                                    \
    inline static TYPE init##NAME()                                                                                    \
    {                                                                                                                  \
        return INIT;                                                                                                   \
    }                                                                                                                  \
    constexpr static auto getPtr(const ct::Indexer<__COUNTER__ - REFLECT_COUNT_BEGIN>)                                 \
    {                                                                                                                  \
        return ct::makeMemberObjectPointer<mo::ParamReflectionFlags::kSOURCE>(                                         \
            #NAME, &DataType::NAME, ct::Initializer<TYPE>(&DataType::init##NAME, #INIT));                              \
    }
