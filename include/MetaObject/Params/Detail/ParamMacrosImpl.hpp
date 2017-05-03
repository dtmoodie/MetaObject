#pragma once
#include <MetaObject/Detail/HelperMacros.hpp>
#include "MetaObject/Params/ParamInfo.hpp"

#include "RuntimeObjectSystem/ISimpleSerializer.h"
#include "cereal/cereal.hpp"

namespace mo {
    template<typename T> struct argument_type;
    template<typename T, typename U> struct argument_type<T(U)> { typedef U type; };
}

#define PARAM_(type_, name, init, N) \
LOAD_SAVE_(name, N) \
INIT_SET_(name, init, N) \
static void list_Param_info_(std::vector<mo::ParamInfo*>& info, mo::_counter_<N> dummy){ \
    static mo::ParamInfo s_info(mo::TypeInfo(typeid(mo::argument_type<void(type_)>::type)), \
                              #name, "", "", mo::Control_e, #init); \
    info.push_back(&s_info); \
    list_Param_info_(info, --dummy); \
} \
SERIALIZE_(name, N)

#define INIT_SET_(name, init, N) \
void init_Params_(bool firstInit, mo::_counter_<N> dummy){ \
    if(firstInit) \
        name = init; \
    name##_param.setMtx(_mtx); \
    name##_param.updatePtr(&name); \
    name##_param.setContext(_ctx); \
    name##_param.setName(#name); \
    addParam(&name##_param); \
    init_Params_(firstInit, --dummy); \
}

#define SET_(name, init, N) \
void init_Params_(bool firstInit, mo::_counter_<N> dummy){ \
    if(firstInit) \
        name = init; \
    init_Params_(firstInit, --dummy); \
}

#define INIT_(name,  N) \
void init_Params_(bool firstInit, mo::_counter_<N> dummy){ \
    name##_param.setMtx(_mtx); \
    name##_param.updatePtr(&name); \
    name##_param.setContext(_ctx); \
    name##_param.setName(#name); \
    addParam(&name##_param); \
    init_Params_(firstInit, --dummy); \
}

#define LOAD_SAVE_(name, N) \
template<class T> void _load_Params(T& ar, mo::_counter_<N> dummy){ \
  _load_Params(ar, --dummy); \
  ar(CEREAL_NVP(name)); \
} \
template<class T> void _save_Params(T& ar, mo::_counter_<N> dummy) const{ \
  _save_Params(ar, --dummy); \
  ar(CEREAL_NVP(name)); \
}

#define ENUM_PARAM_(N, name, ...) \
template<class T> void _serialize_Params(T& ar, mo::_counter_<N> dummy){ \
    _serialize_Params(ar, --dummy); \
    ar(CEREAL_NVP(name)); \
} \
void init_Params_(bool firstInit, mo::_counter_<N> dummy){ \
    if(firstInit){ \
        name.SetValue(ENUM_EXPAND(__VA_ARGS__)); \
    }\
    name##_param.setMtx(_mtx); \
    name##_param.updatePtr(&name); \
    name##_param.setContext(_ctx); \
    name##_param.setName(#name); \
    addParam(&name##_param); \
    init_Params_(firstInit, --dummy); \
} \
static void list_Param_info_(std::vector<mo::ParamInfo*>& info, mo::_counter_<N> dummy){ \
    static mo::ParamInfo s_info(mo::TypeInfo(typeid(mo::EnumParam)), #name); \
    info.push_back(&s_info); \
    list_Param_info_(info, --dummy); \
} \
SERIALIZE_(name, N)



#define OUTPUT_(type_, name, init, N) \
void init_Params_(bool firstInit, mo::_counter_<N> dummy){ \
    if(firstInit) \
        name = name##_param.reset(init); \
    name##_param.setMtx(_mtx); \
    name##_param.updatePtr(&name); \
    name##_param.setContext(_ctx); \
    name##_param.setName(#name); \
    name##_param.setFlags(mo::ParamType::Output_e); \
    addParam(&name##_param); \
    init_Params_(firstInit, --dummy); \
} \
static void list_Param_info_(std::vector<mo::ParamInfo*>& info, mo::_counter_<N> dummy){ \
    static mo::ParamInfo s_info(mo::TypeInfo(typeid(mo::argument_type<void(type_)>::type)), #name, "", "", mo::ParamType::Output_e); \
    info.push_back(&s_info); \
    list_Param_info_(info, --dummy); \
} \
SERIALIZE_(name, N) \
void init_outputs_(mo::_counter_<N> dummy){ \
    name = name##_param.reset(init); \
}


#define TOOLTIP_(NAME, TOOLTIP, N) \
static void list_Param_info_(std::vector<mo::ParamInfo*>& info, mo::_counter_<N> dummy){ \
    list_Param_info_(info, --dummy); \
    for(auto it : info) \
    { \
        if(it->name == #NAME) \
        { \
            if(it->tooltip.empty()) \
            { \
                it->tooltip = TOOLTIP; \
            } \
        } \
    } \
}

#define STATUS_(type_, name, init, N)\
template<class T> void _serialize_Params(T& ar, mo::_counter_<N> dummy){ \
    _serialize_Params(ar, --dummy); \
    ar(CEREAL_NVP(name)); \
} \
void init_Params_(bool firstInit, mo::_counter_<N> dummy){ \
    if(firstInit) \
        name = init; \
    name##_param.setMtx(_mtx); \
    name##_param.updatePtr(&name); \
    name##_param.setContext(_ctx); \
    name##_param.setName(#name); \
    name##_param.setFlags(mo::ParamType::State_e); \
    addParam(&name##_param); \
    init_Params_(firstInit, --dummy); \
} \
static void list_Param_info_(std::vector<mo::ParamInfo*>& info, mo::_counter_<N> dummy){ \
    static mo::ParamInfo s_info(mo::TypeInfo(typeid(mo::argument_type<void(type_)>::type)), #name, "", "", mo::ParamType::State_e); \
    info.push_back(&s_info); \
    list_Param_info_(info, --dummy); \
} \
SERIALIZE_(name, N)

#define SERIALIZE_(name, N) \
void _serialize_Params(ISimpleSerializer* pSerializer, mo::_counter_<N> dummy){ \
    SERIALIZE(name); \
    _serialize_Params(pSerializer, --dummy); \
} 

#define INPUT_PARAM_(type_, name, init, N) \
void init_Params_(bool firstInit, mo::_counter_<N> dummy){ \
    name##_param.setMtx(_mtx); \
    name##_param.SetUserDataPtr(&name); \
    name##_param.setName(#name); \
    addParam(&name##_param); \
    init_Params_(firstInit, --dummy); \
} \
static void list_Param_info_(std::vector<mo::ParamInfo*>& info, mo::_counter_<N> dummy){ \
    static mo::ParamInfo s_info(mo::TypeInfo(typeid(mo::argument_type<void(type_)>::type)), #name, "", "", mo::ParamType::Input_e); \
    info.push_back(&s_info); \
    list_Param_info_(info, --dummy); \
}
