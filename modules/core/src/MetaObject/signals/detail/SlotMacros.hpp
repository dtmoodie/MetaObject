#pragma once

#include <boost/preprocessor/facilities/overload.hpp>
#ifdef _MSC_VER
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/facilities/empty.hpp>
#endif
#include "MetaObject/core/detail/Counter.hpp"
#include "MetaObject/core/detail/forward.hpp"
#include "MetaObject/signals/SlotInfo.hpp"
#include "MetaObject/signals/TSlot.hpp"

// -------------------------------------------------------------------------------------------
#define SLOT_N(NAME, N, RETURN, ...)                                                                                   \
    virtual RETURN NAME(__VA_ARGS__);                                                                                  \
    mo::TSlot<RETURN(__VA_ARGS__)> COMBINE(_slot_##NAME##_, N);                                                        \
    template <class V, class F, class... Args>                                                                         \
    inline void reflectHelper(                                                                                         \
        V& visitor, F visit_filter, mo::MemberFilter<mo::SLOTS> filter, mo::_counter_<N> cnt, Args&&... args)          \
    {                                                                                                                  \
        visitor(mo::tagSlot(COMBINE(_slot_##NAME##_, N)),                                                              \
                mo::tagFunction(static_cast<RETURN (THIS_CLASS::*)(__VA_ARGS__)>(&THIS_CLASS::NAME)),                  \
                mo::Name(#NAME),                                                                                       \
                cnt,                                                                                                   \
                std::forward<Args>(args)...);                                                                          \
        reflectHelper(visitor, visit_filter, filter, --cnt, std::forward<Args>(args)...);                              \
    }                                                                                                                  \
    template <class V, class F, class... Args>                                                                         \
    static inline void reflectHelperStatic(                                                                            \
        V& visitor, F visit_filter, mo::MemberFilter<mo::SLOTS> filter, mo::_counter_<N> cnt, Args&&... args)          \
    {                                                                                                                  \
        visitor(mo::tagFunction(static_cast<RETURN (THIS_CLASS::*)(__VA_ARGS__)>(&THIS_CLASS::NAME)),                  \
                mo::Name(#NAME),                                                                                       \
                cnt,                                                                                                   \
                std::forward<Args>(args)...);                                                                          \
        reflectHelperStatic(visitor, visit_filter, filter, --cnt, std::forward<Args>(args)...);                        \
    }                                                                                                                  \
    template <class Sig>                                                                                               \
    mo::TSlot<RETURN(__VA_ARGS__)>* getSlot_##NAME(                                                                    \
        typename std::enable_if<std::is_same<Sig, RETURN(__VA_ARGS__)>::value>::type* = 0)                             \
    {                                                                                                                  \
        return &COMBINE(_slot_##NAME##_, N);                                                                           \
    }

#define STATIC_SLOT_N(NAME, N, RETURN, ...)                                                                            \
    static RETURN NAME(__VA_ARGS__);                                                                                   \
    template <class V, class... Args>                                                                                  \
    static inline void reflectHelperStatic(V& visitor,                                                                 \
                                           mo::VisitationFilter<mo::LIST> visit_filter,                                \
                                           mo::MemberFilter<mo::SLOTS> filter,                                         \
                                           mo::_counter_<N> cnt,                                                       \
                                           Args&&... args)                                                             \
    {                                                                                                                  \
        visitor(mo::tagStaticFunction(static_cast<RETURN (*)(__VA_ARGS__)>(&THIS_CLASS::NAME)),                        \
                mo::Name(#NAME),                                                                                       \
                cnt,                                                                                                   \
                std::forward<Args>(args)...);                                                                          \
        reflectHelperStatic(visitor, visit_filter, filter, --cnt, std::forward<Args>(args)...);                        \
    }

#define STATIC_SLOT_1(RETURN, N, NAME)                                                                                 \
    static RETURN NAME();                                                                                              \
    template <class V, class... Args>                                                                                  \
    static inline void reflectHelperStatic(V& visitor,                                                                 \
                                           mo::VisitationFilter<mo::LIST> visit_filter,                                \
                                           mo::MemberFilter<mo::SLOTS> filter,                                         \
                                           mo::_counter_<N> cnt,                                                       \
                                           Args&&... args)                                                             \
    {                                                                                                                  \
        visitor(mo::tagStaticFunction(static_cast<RETURN (*)()>(&THIS_CLASS::NAME)),                                   \
                mo::Name(#NAME),                                                                                       \
                cnt,                                                                                                   \
                std::forward<Args>(args)...);                                                                          \
        reflectHelperStatic(visitor, visit_filter, filter, --cnt, std::forward<Args>(args)...);                        \
    }

#define SLOT_1(RETURN, N, NAME)                                                                                        \
    virtual RETURN NAME();                                                                                             \
    mo::TSlot<RETURN(void)> COMBINE(_slot_##NAME##_, N);                                                               \
    template <class V, class F, class... Args>                                                                         \
    inline void reflectHelper(                                                                                         \
        V& visitor, F visit_filter, mo::MemberFilter<mo::SLOTS> filter, mo::_counter_<N> cnt, Args&&... args)          \
    {                                                                                                                  \
        visitor(mo::tagSlot(COMBINE(_slot_##NAME##_, N)),                                                              \
                mo::tagFunction(static_cast<RETURN (THIS_CLASS::*)()>(&THIS_CLASS::NAME)),                             \
                mo::Name(#NAME),                                                                                       \
                cnt,                                                                                                   \
                std::forward<Args>(args)...);                                                                          \
        reflectHelper(visitor, visit_filter, filter, --cnt, std::forward<Args>(args)...);                              \
    }                                                                                                                  \
    template <class V, class F, class... Args>                                                                         \
    static inline void reflectHelperStatic(                                                                            \
        V& visitor, F visit_filter, mo::MemberFilter<mo::SLOTS> filter, mo::_counter_<N> cnt, Args&&... args)          \
    {                                                                                                                  \
        visitor(mo::tagFunction(static_cast<RETURN (THIS_CLASS::*)()>(&THIS_CLASS::NAME)),                             \
                mo::Name(#NAME),                                                                                       \
                cnt,                                                                                                   \
                std::forward<Args>(args)...);                                                                          \
        reflectHelperStatic(visitor, visit_filter, filter, --cnt, std::forward<Args>(args)...);                        \
    }                                                                                                                  \
    template <class Sig>                                                                                               \
    mo::TSlot<RETURN()>* getSlot_##NAME(typename std::enable_if<std::is_same<Sig, RETURN()>::value>::type* = 0)        \
    {                                                                                                                  \
        return &COMBINE(_slot_##NAME##_, N);                                                                           \
    }

#define STATIC_SLOT_2(RETURN, N, NAME, ...) STATIC_SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define STATIC_SLOT_3(RETURN, N, NAME, ...) STATIC_SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define STATIC_SLOT_4(RETURN, N, NAME, ...) STATIC_SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define STATIC_SLOT_5(RETURN, N, NAME, ...) STATIC_SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define STATIC_SLOT_6(RETURN, N, NAME, ...) STATIC_SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define STATIC_SLOT_7(RETURN, N, NAME, ...) STATIC_SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define STATIC_SLOT_8(RETURN, N, NAME, ...) STATIC_SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define STATIC_SLOT_9(RETURN, N, NAME, ...) STATIC_SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define STATIC_SLOT_10(RETURN, N, NAME, ...) STATIC_SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define STATIC_SLOT_11(RETURN, N, NAME, ...) STATIC_SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define STATIC_SLOT_12(RETURN, N, NAME, ...) STATIC_SLOT_N(NAME, N, RETURN, __VA_ARGS__)

#define SLOT_2(RETURN, N, NAME, ...) SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_3(RETURN, N, NAME, ...) SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_4(RETURN, N, NAME, ...) SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_5(RETURN, N, NAME, ...) SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_6(RETURN, N, NAME, ...) SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_7(RETURN, N, NAME, ...) SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_8(RETURN, N, NAME, ...) SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_9(RETURN, N, NAME, ...) SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_10(RETURN, N, NAME, ...) SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_11(RETURN, N, NAME, ...) SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_12(RETURN, N, NAME, ...) SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_13(RETURN, N, NAME, ...) SLOT_N(NAME, N, RETURN, __VA_ARGS__)

#define DESCRIBE_SLOT_(NAME, DESCRIPTION, N)                                                                           \
    std::string _slot_description_by_name(const std::string& name, mo::_counter_<N> dummy)                             \
    {                                                                                                                  \
        (void)dummy;                                                                                                   \
        if (name == #NAME)                                                                                             \
            return DESCRIPTION;                                                                                        \
    }                                                                                                                  \
    std::vector<slot_info> _list_slots(mo::_counter_<N> dummy)                                                         \
    {                                                                                                                  \
        (void)dummy;                                                                                                   \
        auto slot_info = _list_slots(mo::_counter_<N - 1>());                                                          \
        for (auto& info : slot_info)                                                                                   \
        {                                                                                                              \
            if (info.name == #NAME)                                                                                    \
            {                                                                                                          \
                info.description = DESCRIPTION;                                                                        \
            }                                                                                                          \
        }                                                                                                              \
        return slot_info;                                                                                              \
    }

#define SLOT_TOOLTIP_(name, tooltip, N)                                                                                \
    static void _list_slots(std::vector<mo::SlotInfo*>& info, mo::_counter_<N> dummy)                                  \
    {                                                                                                                  \
        _list_slots(info, --dummy);                                                                                    \
        for (auto it : info)                                                                                           \
        {                                                                                                              \
            if (it->name == #name)                                                                                     \
            {                                                                                                          \
                if (it->tooltip.empty())                                                                               \
                {                                                                                                      \
                    it->tooltip = tooltip;                                                                             \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }

#ifndef __CUDACC__

#ifdef _MSC_VER

#define MO_SLOT(RET, ...)                                                                                              \
    BOOST_PP_CAT(BOOST_PP_OVERLOAD(SLOT_, __VA_ARGS__)(RET, __COUNTER__, __VA_ARGS__), BOOST_PP_EMPTY())

#define MO_STATIC_SLOT(RET, ...)                                                                                       \
    BOOST_PP_CAT(BOOST_PP_OVERLOAD(STATIC_SLOT_, __VA_ARGS__)(RET, __COUNTER__, __VA_ARGS__), BOOST_PP_EMPTY())

#else // _MSC_VER

#define MO_SLOT(NAME, ...)                                                                                             \
    BOOST_PP_OVERLOAD(SLOT_, __VA_ARGS__)                                                                              \
    (NAME, __COUNTER__, __VA_ARGS__)

#define MO_STATIC_SLOT(NAME, ...)                                                                                      \
    BOOST_PP_OVERLOAD(STATIC_SLOT_, __VA_ARGS__)                                                                       \
    (NAME, __COUNTER__, __VA_ARGS__)

#endif // _MSC_VER

#define DESCRIBE_SLOT(NAME, DESCRIPTION) DESCRIBE_SLOT_(NAME, DESCRIPTION, __COUNTER__)

#else // __CUDAACC__

#define MO_SLOT(RET, ...)

#define MO_STATIC_SLOT(RET, ...)
#define DESCRIBE_SLOT(NAME, DESCRIPTION)
#endif
#define PARAM_UPDATE_SLOT(NAME)                                                                                        \
    MO_SLOT(void,                                                                                                      \
            on_##NAME##_modified,                                                                                      \
            mo::IParam*,                                                                                               \
            mo::Context*,                                                                                              \
            mo::OptionalTime,                                                                                        \
            size_t,                                                                                                    \
            const std::shared_ptr<mo::ICoordinateSystem>&,                                                             \
            mo::UpdateFlags)
#define SLOT_TOOLTIP(name, tooltip) SLOT_TOOLTIP_(name, tooltip, __COUNTER__)
