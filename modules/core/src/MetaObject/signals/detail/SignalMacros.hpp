#pragma once
#include <boost/preprocessor/facilities/overload.hpp>
#ifdef _MSC_VER
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/facilities/empty.hpp>
#endif

#include "MetaObject/core/detail/HelperMacros.hpp"
#include "MetaObject/signals/SignalInfo.hpp"
#include "MetaObject/signals/TSignal.hpp"
#include <vector>

#define SIGNAL_CALL_1(N, name, ret)                                                                                    \
    inline ret sig_##name()                                                                                            \
    {                                                                                                                  \
        return COMBINE(_sig_##name##_, N)();                                                                           \
    }

#define SIGNAL_CALL_2(N, name, ret, ARG1)                                                                              \
    inline ret sig_##name(ARG1 const& arg1)                                                                            \
    {                                                                                                                  \
        return COMBINE(_sig_##name##_, N)(arg1);                                                                       \
    }

#define SIGNAL_CALL_3(N, name, ret, ARG1, ARG2)                                                                        \
    inline ret sig_##name(ARG1 const& arg1, ARG2 const& arg2)                                                          \
    {                                                                                                                  \
        return COMBINE(_sig_##name##_, N)(arg1, arg2);                                                                 \
    }

#define SIGNAL_CALL_4(N, name, ret, ARG1, ARG2, ARG3)                                                                  \
    inline ret sig_##name(ARG1 const& arg1, ARG2 const& arg2, ARG3 const& arg3)                                        \
    {                                                                                                                  \
        return COMBINE(_sig_##name##_, N)(arg1, arg2, arg3);                                                           \
    }

#define SIGNAL_CALL_5(N, name, ret, ARG1, ARG2, ARG3, ARG4)                                                            \
    inline ret sig_##name(ARG1 const& arg1, ARG2 const& arg2, ARG3 const& arg3, ARG4 const& arg4)                      \
    {                                                                                                                  \
        return COMBINE(_sig_##name##_, N)(arg1, arg2, arg3, arg4);                                                     \
    }

#define SIGNAL_CALL_6(N, name, ret, ARG1, ARG2, ARG3, ARG4, ARG5)                                                      \
    inline ret sig_##name(ARG1 const& arg1, ARG2 const& arg2, ARG3 const& arg3, ARG4 const& arg4, ARG5 const& arg5)    \
    {                                                                                                                  \
        return COMBINE(_sig_##name##_, N)(arg1, arg2, arg3, arg4, arg5);                                               \
    }

#define SIGNAL_CALL_7(N, name, ret, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6)                                                \
    inline ret sig_##name(                                                                                             \
        ARG1 const& arg1, ARG2 const& arg2, ARG3 const& arg3, ARG4 const& arg4, ARG5 const& arg5, ARG6 const& arg6)    \
    {                                                                                                                  \
        return COMBINE(_sig_##name##_, N)(arg1, arg2, arg3, arg4, arg5, arg6);                                         \
    }

#define SIGNAL_CALL_8(N, name, ret, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7)                                          \
    inline ret sig_##name(ARG1 const& arg1,                                                                            \
                          ARG2 const& arg2,                                                                            \
                          ARG3 const& arg3,                                                                            \
                          ARG4 const& arg4,                                                                            \
                          ARG5 const& arg5,                                                                            \
                          ARG6 const& arg6,                                                                            \
                          ARG7 const& arg7)                                                                            \
    {                                                                                                                  \
        return COMBINE(_sig_##name##_, N)(arg1, arg2, arg3, arg4, arg5, arg6, arg7);                                   \
    }

#define SIGNAL_CALL_9(N, name, ret, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8)                                    \
    inline ret sig_##name(ARG1 const& arg1,                                                                            \
                          ARG2 const& arg2,                                                                            \
                          ARG3 const& arg3,                                                                            \
                          ARG4 const& arg4,                                                                            \
                          ARG5 const& arg5,                                                                            \
                          ARG6 const& arg6,                                                                            \
                          ARG7 const& arg7,                                                                            \
                          ARG8 const& arg8)                                                                            \
    {                                                                                                                  \
        return COMBINE(_sig_##name##_, N)(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);                             \
    }

#ifdef _MSC_VER

#define SIGNAL_CALL(N, name, ...)                                                                                      \
    CT_PP_CAT(CT_PP_OVERLOAD(SIGNAL_CALL_, __VA_ARGS__)(N, name, __VA_ARGS__), CT_PP_EMPTY())

#else

#define SIGNAL_CALL(N, name, ...) CT_PP_OVERLOAD(SIGNAL_CALL_, __VA_ARGS__)(N, name, __VA_ARGS__)

#endif

#define MO_SIGNAL_(N, RETURN, NAME, ...)                                                                               \
    mo::TSignal<RETURN(__VA_ARGS__)> COMBINE(_sig_##NAME##_, N);                                                       \
    SIGNAL_CALL(N, NAME, RETURN, ##__VA_ARGS__)                                                                        \
    constexpr static auto getPtr(const ct::Indexer<__COUNTER__ - REFLECT_COUNT_START>)                                 \
    {                                                                                                                  \
        return ct::makeMemberObjectPointer(#NAME, &DataType::COMBINE(_sig_##NAME##_, N));                              \
    }

#define DESCRIBE_SIGNAL_(NAME, DESCRIPTION, N)                                                                         \
    std::vector<slot_info> _list_signals(const ct::Indexer<N> dummy)                                                   \
    {                                                                                                                  \
        auto signal_info = _list_signals(--dummy);                                                                     \
        for (auto& info : signal_info)                                                                                 \
        {                                                                                                              \
            if (info.name == #NAME)                                                                                    \
            {                                                                                                          \
                info.description = DESCRIPTION;                                                                        \
            }                                                                                                          \
        }                                                                                                              \
        return signal_info;                                                                                            \
    }

#ifndef __CUDACC__
#define DESCRIBE_SIGNAL(NAME, DESCRIPTION) DESCRIBE_SIGNAL_(NAME, DESCRIPTION, __COUNTER__);
#define MO_SIGNAL(RETURN, NAME, ...) MO_SIGNAL_(__LINE__, RETURN, NAME, ##__VA_ARGS__)
#else
#define DESCRIBE_SIGNAL(NAME, DESCRIPTION)
#define MO_SIGNAL(RETURN, NAME, ...)
#endif
