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
    template <class T>                                                                                                 \
    void _serialize_params(T& ar, mo::_counter_<N> dummy)                                                              \
    {                                                                                                                  \
        _serialize_params(ar, --dummy);                                                                                \
        ar(CEREAL_NVP(name));                                                                                          \
    }                                                                                                                  \
    void _init_params(bool firstInit, mo::_counter_<N> dummy)                                                          \
    {                                                                                                                  \
        if (firstInit)                                                                                                 \
        {                                                                                                              \
            name.setValue(ENUM_EXPAND(__VA_ARGS__));                                                                   \
        }                                                                                                              \
        name##_param.setName(#name);                                                                                   \
        addParam(&name##_param);                                                                                       \
        name##_param.updatePtr(&name);                                                                                 \
        _init_params(firstInit, --dummy);                                                                              \
    }                                                                                                                  \
    static void _list_param_info(std::vector<mo::ParamInfo*>& info, mo::_counter_<N> dummy)                            \
    {                                                                                                                  \
        static mo::ParamInfo s_info(mo::TypeInfo(typeid(mo::EnumParam)), #name);                                       \
        info.push_back(&s_info);                                                                                       \
        _list_param_info(info, --dummy);                                                                               \
    }                                                                                                                  \
    SERIALIZE_(name, N)

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

#define TOOLTIP_(NAME, TOOLTIP, N)                                                                                     \
    static void _list_param_info(std::vector<mo::ParamInfo*>& info, mo::_counter_<N> dummy)                            \
    {                                                                                                                  \
        _list_param_info(info, --dummy);                                                                               \
        for (auto it : info)                                                                                           \
        {                                                                                                              \
            if (it->name == #NAME)                                                                                     \
            {                                                                                                          \
                if (it->tooltip.empty())                                                                               \
                {                                                                                                      \
                    it->tooltip = TOOLTIP;                                                                             \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }

#define STATUS_(type_, name, init, N)                                                                                  \
    template <class T>                                                                                                 \
    void _serialize_params(T& ar, mo::_counter_<N> dummy)                                                              \
    {                                                                                                                  \
        _serialize_Params(ar, --dummy);                                                                                \
        ar(CEREAL_NVP(name));                                                                                          \
    }                                                                                                                  \
    void _init_params(bool firstInit, mo::_counter_<N> dummy)                                                          \
    {                                                                                                                  \
        if (firstInit)                                                                                                 \
            name = init;                                                                                               \
        name##_param.setName(#name);                                                                                   \
        addParam(&name##_param);                                                                                       \
        name##_param.updatePtr(&name);                                                                                 \
        name##_param.setFlags(mo::ParamFlags::State_e);                                                                \
        _init_params(firstInit, --dummy);                                                                              \
    }                                                                                                                  \
    static void _list_param_info(std::vector<mo::ParamInfo*>& info, mo::_counter_<N> dummy)                            \
    {                                                                                                                  \
        static mo::ParamInfo s_info(                                                                                   \
            mo::TypeInfo(typeid(mo::argument_type<void(type_)>::type)), #name, "", "", mo::ParamFlags::State_e);       \
        info.push_back(&s_info);                                                                                       \
        _list_param_info(info, --dummy);                                                                               \
    }                                                                                                                  \
    SERIALIZE_(name, N)

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
