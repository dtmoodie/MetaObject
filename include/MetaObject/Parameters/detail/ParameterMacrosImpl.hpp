#pragma once


#define PARAM__(type, name, init, N) \
void init_parameters_(bool firstInit, mo::_counter_<N> dummy) \
{ \
    if(firstInit) \
        name = init; \
    if(name##_param == nullptr) \
    { \
        name##_param.reset(new mo::TypedParameterPtr<type>(#name, &name, mo::Control_e, false)); \
        name##_param->SetContext(_ctx); \
    }else \
    { \
        name##_param->UpdateData(&name); \
        name##_param->SetContext(_ctx); \
    } \
    init_parameters_(firstInit, --dummy); \
}