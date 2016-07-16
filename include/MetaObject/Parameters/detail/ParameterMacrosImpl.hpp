#pragma once


#define PARAM__(type, name, init, N) \
void init_parameters_(bool firstInit, mo::_counter_<N> dummy) \
{ \
    if(firstInit) \
        name = init; \
    name##_param.UpdateData(&name); \
    name##_param.SetContext(_ctx); \
    name##_param.SetName(#name); \
    AddParameter(&name##_param); \
    init_parameters_(firstInit, --dummy); \
}
#define INPUT_PARAM__(type, name, init, N) \
void init_parameters_(bool firstInit, mo::_counter_<N> dummy) \
{ \
    name##_param.SetUserDataPtr(&name); \
    AddParameter(&name##_param); \
    init_parameters_(firstInit, --dummy); \
}