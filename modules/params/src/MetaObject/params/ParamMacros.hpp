#pragma once

#include "MetaObject/core/detail/HelperMacros.hpp"
#include "MetaObject/params/IParam.hpp"
#include "MetaObject/params/TInputParam.hpp"
#include "MetaObject/params/TParamPtr.hpp"
#include "MetaObject/params/detail/ParamMacrosImpl.hpp"
#include "MetaObject/types/file_types.hpp"

namespace mo
{
    enum ParamReflectionFlags : ct::Flag_t
    {
        kCONTROL = 1 << (ct::CT_RESERVED_FLAG_BITS + 1),
        kSTATE = kCONTROL << 1,
        kSTATUS = kSTATE << 1,
        kINPUT = kSTATUS << 1,
        kOUTPUT = kINPUT << 1,

        kOPTIONAL = kOUTPUT << 1,
        kSOURCE = kOPTIONAL << 1,

        kSIGNAL = kSOURCE << 1,
        kSLOT = kSIGNAL << 1
    };
}

#define REFLECT_INTERNAL_WITH_FLAG(FLAGS, TYPE, NAME, ...)                                                             \
    TYPE NAME = __VA_ARGS__;                                                                                           \
    constexpr static auto getPtr(const ct::Indexer<__COUNTER__ - REFLECT_COUNT_START>)                                 \
    {                                                                                                                  \
        return ct::makeMemberObjectPointer<FLAGS>(#NAME, &DataType::NAME);                                             \
    }

#define PARAM(TYPE, NAME, ...)                                                                                         \
    REFLECT_INTERNAL_MEMBER(mo::TParamPtr<TYPE>, NAME##_param)                                                         \
    REFLECT_INTERNAL_WITH_FLAG(mo::kCONTROL, TYPE, NAME, __VA_ARGS__)

#define ENUM_PARAM(name, ...)                                                                                          \
    mo::TParamPtr<mo::EnumParam> name##_param;                                                                         \
    mo::EnumParam name;                                                                                                \
    ENUM_PARAM_(__COUNTER__, name, __VA_ARGS__)

#define RANGED_PARAM(type, name, init, min, max)

#define INPUT(TYPE, NAME)                                                                                              \
    REFLECT_INTERNAL_WITH_FLAG(mo::kOUTPUT, mo::TInputParamPtr<TYPE>::Input_t, NAME, nullptr)                          \
    REFLECT_INTERNAL_MEMBER(mo::TInputParamPtr<TYPE>, NAME##_param)

#define OPTIONAL_INPUT(type, name)                                                                                     \
    REFLECT_INTERNAL_WITH_FLAGS(mo::kOUTPUT | mo::kOPTIONAL, mo::TInputParamPtr<type_>::Input_t, nullptr)              \
    REFLECT_INTERNAL(mo::TInputParamPtr<TYPE>, name##_param)

#define STATE(TYPE, NAME, ...)                                                                                         \
    REFLECT_INTERNAL_MEMBER(mo::TParamPtr<TYPE>, NAME##_param)                                                         \
    REFLECT_INTERNAL_WITH_FLAG(mo::kSTATE, TYPE, NAME, __VA_ARGS__)

#define STATUS(TYPE, NAME, ...)                                                                                        \
    REFLECT_INTERNAL_MEMBER(mo::TParamPtr<TYPE>, NAME##_param)                                                         \
    REFLECT_INTERNAL_WITH_FLAG(mo::kSTATUS, TYPE, NAME, __VA_ARGS__)

#define OUTPUT(TYPE, NAME, ...)                                                                                        \
    REFLECT_INTERNAL_MEMBER(mo::TParamOutput<TYPE>, NAME##_param)                                                      \
    REFLECT_INTERNAL_WITH_FLAG(mo::kOUTPUT, TYPE, NAME, __VA_ARGS__)

#define SOURCE(type, name, ...)                                                                                        \
    REFLECT_INTERNAL_MEMBER(mo::TParamOutput<TYPE>, NAME##_param)                                                      \
    REFLECT_INTERNAL_WITH_FLAG(mo::kSOURCE, TYPE, NAME, __VA_ARGS__)
