#pragma once
#include "MetaObject/Parameters/ParameterInfo.hpp"
#include "ISimpleSerializer.h"
#include "cereal/cereal.hpp"

#define PARAM_(type, name, init, N) \
template<class T> void _serialize_parameters(T& ar, mo::_counter_<N> dummy) \
{ \
    _serialize_parameters(ar, --dummy); \
    ar(CEREAL_NVP(name)); \
} \
void init_parameters_(bool firstInit, mo::_counter_<N> dummy) \
{ \
    if(firstInit) \
        name = init; \
    name##_param.UpdatePtr(&name); \
    name##_param.SetContext(_ctx); \
    name##_param.SetName(#name); \
    AddParameter(&name##_param); \
    init_parameters_(firstInit, --dummy); \
} \
static void list_parameter_info_(std::vector<mo::ParameterInfo*>& info, mo::_counter_<N> dummy) \
{ \
    static mo::ParameterInfo s_info(mo::TypeInfo(typeid(type)), #name); \
    info.push_back(&s_info); \
    list_parameter_info_(info, --dummy); \
} \
SERIALIZE_(name, N)

#define OUTPUT_(type, name, init, N) \
void init_parameters_(bool firstInit, mo::_counter_<N> dummy) \
{ \
    if(firstInit) \
        name = init; \
    name##_param.UpdatePtr(&name); \
    name##_param.SetContext(_ctx); \
    name##_param.SetName(#name); \
    name##_param.SetFlags(mo::ParameterType::Output_e); \
    AddParameter(&name##_param); \
    init_parameters_(firstInit, --dummy); \
} \
static void list_parameter_info_(std::vector<mo::ParameterInfo*>& info, mo::_counter_<N> dummy) \
{ \
    static mo::ParameterInfo s_info(mo::TypeInfo(typeid(type)), #name, "", "", mo::ParameterType::Output_e); \
    info.push_back(&s_info); \
    list_parameter_info_(info, --dummy); \
} \
SERIALIZE_(name, N)


#define TOOLTIP_(NAME, TOOLTIP, N) \
static void list_parameter_info_(std::vector<mo::ParameterInfo*>& info, mo::_counter_<N> dummy) \
{ \
    list_parameter_info_(info, --dummy); \
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

#define STATUS_(type, name, init, N)\
template<class T> void _serialize_parameters(T& ar, mo::_counter_<N> dummy) \
{ \
    _serialize_parameters(ar, --dummy); \
    ar(CEREAL_NVP(name)); \
} \
void init_parameters_(bool firstInit, mo::_counter_<N> dummy) \
{ \
    if(firstInit) \
        name = init; \
    name##_param.UpdatePtr(&name); \
    name##_param.SetContext(_ctx); \
    name##_param.SetName(#name); \
    name##_param.SetFlags(mo::ParameterType::State_e); \
    AddParameter(&name##_param); \
    init_parameters_(firstInit, --dummy); \
} \
SERIALIZE_(name, N)

#define SERIALIZE_(name, N) \
void _serialize_parameters(ISimpleSerializer* pSerializer, mo::_counter_<N> dummy) \
{ \
    SERIALIZE(name); \
} 

#define INPUT_PARAM_(type, name, init, N) \
void init_parameters_(bool firstInit, mo::_counter_<N> dummy) \
{ \
    name##_param.SetUserDataPtr(&name); \
    name##_param.SetName(#name); \
    AddParameter(&name##_param); \
    init_parameters_(firstInit, --dummy); \
}

