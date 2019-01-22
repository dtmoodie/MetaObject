#pragma once
#include "MetaObject/params/ParamInfo.hpp"
#include <MetaObject/core/detail/HelperMacros.hpp>

#include "RuntimeObjectSystem/ISimpleSerializer.h"
#include "cereal/cereal.hpp"

namespace mo
{
    template <typename T>
    struct argument_type;
    template <typename T, typename U>
    struct argument_type<T(U)>
    {
        typedef U type;
    };
}

#define VISIT(NAME, TYPE, ...)                                                                                         \
    template <class V, class... Args, mo::VisitationType FILTER>                                                       \
    inline void reflectHelper(V& visitor,                                                                              \
                              mo::VisitationFilter<FILTER> filter,                                                     \
                              mo::MemberFilter<TYPE> param,                                                            \
                              const ct::Indexer<__COUNTER__> cnt,                                                          \
                              Args&&... args)                                                                          \
    {                                                                                                                  \
        visitor(mo::tagData(&NAME), mo::Name(#NAME), mo::tagParam(NAME##_param), cnt, std::forward<Args>(args)...);    \
        if (FILTER == mo::INIT)                                                                                        \
            NAME = __VA_ARGS__;                                                                                        \
        reflectHelper(visitor, filter, param, --cnt, std::forward<Args>(args)...);                                     \
    }                                                                                                                  \
    template <class V, class... Args, mo::VisitationType FILTER>                                                       \
    static inline void reflectHelperStatic(V& visitor,                                                                 \
                                           mo::VisitationFilter<FILTER> filter,                                        \
                                           mo::MemberFilter<TYPE> param,                                               \
                                           const ct::Indexer<__COUNTER__> cnt,                                             \
                                           Args&&... args)                                                             \
    {                                                                                                                  \
        visitor(mo::Name(#NAME),                                                                                       \
                mo::tagType<typename std::decay<decltype(THIS_CLASS::NAME)>::type>(),                                  \
                cnt,                                                                                                   \
                std::forward<Args>(args)...);                                                                          \
        reflectHelperStatic(visitor, filter, param, --cnt, std::forward<Args>(args)...);                               \
    }

#define ENUM_PARAM_(N, NAME, ...)                                                                                      \
    template <class V, class... Args, mo::VisitationType FILTER>                                                       \
    inline void reflectHelper(V& visitor,                                                                              \
                              mo::VisitationFilter<FILTER> filter,                                                     \
                              mo::MemberFilter<mo::CONTROL> param,                                                     \
                              const ct::Indexer<__COUNTER__> cnt,                                                          \
                              Args&&... args)                                                                          \
    {                                                                                                                  \
        visitor(mo::tagData(&NAME), mo::Name(#NAME), mo::tagParam(NAME##_param), cnt, std::forward<Args>(args)...);    \
        if (FILTER == mo::INIT)                                                                                        \
        {                                                                                                              \
            NAME.setValue(ENUM_EXPAND(__VA_ARGS__));                                                                   \
        }                                                                                                              \
        reflectHelper(visitor, filter, param, --cnt, std::forward<Args>(args)...);                                     \
    }                                                                                                                  \
    template <class V, class... Args, mo::VisitationType FILTER>                                                       \
    static inline void reflectHelperStatic(V& visitor,                                                                 \
                                           mo::VisitationFilter<FILTER> filter,                                        \
                                           mo::MemberFilter<mo::CONTROL> param,                                        \
                                           const ct::Indexer<__COUNTER__> cnt,                                             \
                                           Args&&... args)                                                             \
    {                                                                                                                  \
        visitor(mo::Name(#NAME),                                                                                       \
                mo::tagType<typename std::decay<decltype(THIS_CLASS::NAME)>::type>(),                                  \
                cnt,                                                                                                   \
                std::forward<Args>(args)...);                                                                          \
        reflectHelperStatic(visitor, filter, param, --cnt, std::forward<Args>(args)...);                               \
    }

#define TOOLTIP_(NAME, TOOLTIP_, N)                                                                                    \
    template <class C, class V, class... Args>                                                                         \
    static inline void reflectHelper(                                                                                  \
        C* , V& visitor, mo::VisitationFilter<mo::TOOLTIP> filter, const ct::Indexer<__COUNTER__> cnt, Args&&... args)  \
    {                                                                                                                  \
        visitor(mo::Name(#NAME), mo::Tooltip(TOOLTIP_), cnt, std::forward<Args>(args)...);                             \
        reflectHelper(visitor, filter, --cnt, std::forward<Args>(args)...);                                            \
    }

#define STATUS_(type_, name, init, N) VISIT(name, mo::STATUS, init)
