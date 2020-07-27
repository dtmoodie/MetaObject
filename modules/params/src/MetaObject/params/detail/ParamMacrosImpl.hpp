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
} // namespace mo

#define TOOLTIP_(NAME, TOOLTIP_, N)                                                                                    \
    template <class C, class V, class... Args>                                                                         \
    static inline void reflectHelper(                                                                                  \
        C*, V& visitor, mo::VisitationFilter<mo::TOOLTIP> filter, const ct::Indexer<__COUNTER__> cnt, Args&&... args)  \
    {                                                                                                                  \
        visitor(mo::Name(#NAME), mo::Tooltip(TOOLTIP_), cnt, std::forward<Args>(args)...);                             \
        reflectHelper(visitor, filter, --cnt, std::forward<Args>(args)...);                                            \
    }

#define STATUS_(type_, name, init, N) VISIT(name, mo::STATUS, init)
