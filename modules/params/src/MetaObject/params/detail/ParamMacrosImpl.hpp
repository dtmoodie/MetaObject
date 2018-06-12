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

#define VISIT(NAME, TYPE)                                                                                            \
    template<class V, class ... Args, mo::VisitationType FILTER>                                                                                  \
    inline void                                                                                                        \
    reflectHelper(V& visitor, mo::VisitationFilter<FILTER> filter, mo::MemberFilter<TYPE> param, mo::_counter_<__COUNTER__> cnt, Args&&... args)     \
    {                                                                                                                  \
        visitor(mo::tagData(&NAME), mo::Name(#NAME), mo::tagParam(NAME##_param), cnt, std::forward<Args>(args)...);    \
        reflectHelper(visitor, filter, param, --cnt, std::forward<Args>(args)...);                                            \
    }


#define PARAM_(type_, name, N, ...)                                                                                    \
    LOAD_SAVE_(name, N)                                                                                                \
    INIT_SET_(name, N, __VA_ARGS__)                                                                                    \
    static void _list_param_info(std::vector<mo::ParamInfo*>& info, mo::_counter_<N> dummy)                            \
    {                                                                                                                  \
        static mo::ParamInfo s_info(mo::TypeInfo(typeid(mo::argument_type<void(type_)>::type)),                        \
                                    #name,                                                                             \
                                    "",                                                                                \
                                    "",                                                                                \
                                    mo::ParamFlags::Control_e,                                                         \
                                    #__VA_ARGS__);                                                                     \
        info.push_back(&s_info);                                                                                       \
        _list_param_info(info, --dummy);                                                                               \
    }                                                                                                                  \
    SERIALIZE_(name, N)

#define INIT_SET_(name, N, ...)                                                                                        \
    void _init_params(bool firstInit, mo::_counter_<N> dummy)                                                          \
    {                                                                                                                  \
        if (firstInit)                                                                                                 \
            name = __VA_ARGS__;                                                                                        \
        name##_param.setName(#name);                                                                                   \
        addParam(&name##_param);                                                                                       \
        name##_param.updatePtr(&name);                                                                                 \
        _init_params(firstInit, --dummy);                                                                              \
    }

#define SET_(name, N, ...)                                                                                             \
    void _init_params(bool firstInit, mo::_counter_<N> dummy)                                                          \
    {                                                                                                                  \
        if (firstInit)                                                                                                 \
            name = __VA_ARGS__;                                                                                        \
        _init_params(firstInit, --dummy);                                                                              \
    }

#define INIT_(name, N)                                                                                                 \
    void _init_params(bool firstInit, mo::_counter_<N> dummy)                                                          \
    {                                                                                                                  \
        name##_param.setName(#name);                                                                                   \
        addParam(&name##_param);                                                                                       \
        name##_param.updatePtr(&name);                                                                                 \
        _init_params(firstInit, --dummy);                                                                              \
    }

#define LOAD_SAVE_(name, N)                                                                                            \
    template <class T>                                                                                                 \
    void _load_params(T& ar, mo::_counter_<N> dummy)                                                                   \
    {                                                                                                                  \
        _load_params(ar, --dummy);                                                                                     \
        ar(CEREAL_NVP(name));                                                                                          \
    }                                                                                                                  \
    template <class T>                                                                                                 \
    void _save_params(T& ar, mo::_counter_<N> dummy) const                                                             \
    {                                                                                                                  \
        _save_params(ar, --dummy);                                                                                     \
        ar(CEREAL_NVP(name));                                                                                          \
    }

#define ENUM_PARAM_(N, name, ...)                                                                                      \
    VISIT(name, mo::CONTROL)                                                                                          \
    template<class V, class ... Args> inline void                                                                       \
    reflectHelper(V& visitor, mo::VisitationFilter<mo::INIT> f, mo::MemberFilter<mo::CONTROL> mem_f, mo::_counter_<N> cnt, Args&& ... args){ \
        name.setValue(ENUM_EXPAND(__VA_ARGS__));                                                                   \
        reflectHelper(visitor, f, mem_f, --cnt, args...); \
    }


#define OUTPUT_(type_, name, init, N)                                                                                  \
    void _init_params(bool firstInit, mo::_counter_<N> dummy)                                                          \
    {                                                                                                                  \
        name##_param.setName(#name);                                                                                   \
        addParam(&name##_param);                                                                                       \
        name##_param.updatePtr(&name);                                                                                 \
        _init_params(firstInit, --dummy);                                                                              \
    }                                                                                                                  \
    static void _list_param_info(std::vector<mo::ParamInfo*>& info, mo::_counter_<N> dummy)                            \
    {                                                                                                                  \
        static mo::ParamInfo s_info(                                                                                   \
            mo::TypeInfo(typeid(mo::argument_type<void(type_)>::type)), #name, "", "", mo::ParamFlags::Output_e);      \
        info.push_back(&s_info);                                                                                       \
        _list_param_info(info, --dummy);                                                                               \
    }                                                                                                                  \
    SERIALIZE_(name, N)                                                                                                \
    void _init_outputs(mo::_counter_<N> dummy) { _init_outputs(--dummy); }

#define TOOLTIP_(NAME, TOOLTIP_, N)                                                                                     \
    template<class V, class ... Args> inline void                                                                       \
    reflectHelper(V& visitor, mo::VisitationFilter<mo::TOOLTIP> filter, mo::_counter_<__COUNTER__> cnt, Args&&... args) \
    {                                                                                                                   \
        visitor(mo::Name(#NAME), mo::Tooltip(TOOLTIP_), cnt, std::forward<Args>(args)...);                              \
        reflectHelper(visitor, filter, --cnt, std::forward<Args>(args)...);                                             \
    }

#define STATUS_(type_, name, init, N) VISIT(name, mo::STATUS)

#define SERIALIZE_(name, N)                                                                                            \
    void _serialize_params(ISimpleSerializer* pSerializer, mo::_counter_<N> dummy)                                     \
    {                                                                                                                  \
        SERIALIZE(name);                                                                                               \
        _serialize_params(pSerializer, --dummy);                                                                       \
    }

#define INPUT_PARAM_(type_, name, init, N)                                                                             \
    void _init_params(bool firstInit, mo::_counter_<N> dummy)                                                          \
    {                                                                                                                  \
        name##_param.setName(#name);                                                                                   \
        addParam(&name##_param);                                                                                       \
        name##_param.SetUserDataPtr(&name);                                                                            \
        _init_params(firstInit, --dummy);                                                                              \
    }                                                                                                                  \
    static void _list_param_info(std::vector<mo::ParamInfo*>& info, mo::_counter_<N> dummy)                            \
    {                                                                                                                  \
        static mo::ParamInfo s_info(                                                                                   \
            mo::TypeInfo(typeid(mo::argument_type<void(type_)>::type)), #name, "", "", mo::ParamType::Input_e);        \
        info.push_back(&s_info);                                                                                       \
        _list_param_info(info, --dummy);                                                                               \
    }
