#pragma once

#include <boost/preprocessor.hpp>
#include "MetaObject/Signals/SignalManager.hpp"
#include "MetaObject/Signals/SignalInfo.hpp"

#define COMBINE1(X,Y) X##Y  // helper macro
#define COMBINE(X,Y) COMBINE1(X,Y)

#define SIGNAL_1(name, N) \
mo::TypedSignal<void(void)>* COMBINE(_sig_##name##_,N) = nullptr; \
inline void sig_##name()\
{\
	if(!_sig_manager) _sig_manager = SignalManager::Instance(); \
	if(COMBINE(_sig_##name##_,N) == nullptr)\
	{ \
		COMBINE(_sig_##name##_,N) = _sig_manager->GetSignal<void(void)>(#name); \
	} \
	(*COMBINE(_sig_##name##_,N))(); \
} \
static std::vector<mo::SignalInfo*> list_signals_(mo::_counter_<N> dummy) \
{ \
    static mo::SignalInfo info{TypeInfo(typeid(void(void))), std::string(#name)}; \
    auto signal_info = list_signals_(--dummy); \
    signal_info.push_back(&info); \
    return signal_info; \
}\


#define SIGNAL_2(name, ARG1, N) \
mo::TypedSignal<void(ARG1)>* COMBINE(_sig_##name##_,N) = nullptr; \
inline void sig_##name(ARG1 arg1)\
{\
	if(!_sig_manager) _sig_manager = SignalManager::Instance(); \
	if(COMBINE(_sig_##name##_,N) == nullptr)\
	{ \
		COMBINE(_sig_##name##_,N) = _sig_manager->GetSignal<void(ARG1)>(#name); \
	} \
	(*COMBINE(_sig_##name##_,N))(arg1); \
}


#define SIGNAL_3(name, ARG1, ARG2,  N) \
mo::TypedSignal<void(ARG1, ARG2)>* COMBINE(_sig_##name##_,N) = nullptr; \
inline void sig_##name(ARG1 arg1, ARG2 arg2)\
{\
	if(!_sig_manager) _sig_manager = SignalManager::Instance(); \
	if(COMBINE(_sig_##name##_,N) == nullptr)\
	{ \
        COMBINE(_sig_##name##_,N) = _sig_manager->GetSignal<void(ARG1, ARG2)>(#name); \
	} \
	(*COMBINE(_sig_##name##_,N))(arg1, arg2); \
} \
static std::vector<SlotInfo*> list_signals_(mo::_counter_<N> dummy) \
{ \
    static SlotInfo info = {TypeInfo(typeid(void(ARG1, ARG2))), std::string(#name)}; \
    auto signal_info = list_signals_(--dummy); \
    signal_info.push_back(&info); \
    return signal_info; \
}\

#define SIGNAL_4(name, ARG1, ARG2, ARG3, N) \
mo::TypedSignal<void(ARG1, ARG2, ARG3)>* COMBINE(_sig_##name##_,N) = nullptr; \
inline void sig_##name(ARG1 arg1, ARG2 arg2, ARG3 arg3)\
{\
	if(!_sig_manager) _sig_manager = SignalManager::Instance(); \
	if(COMBINE(_sig_##name##_,N) == nullptr)\
	{ \
        COMBINE(_sig_##name##_,N) = _sig_manager->GetSignal<void(ARG1, ARG2, ARG3)>(#name); \
	} \
	(*COMBINE(_sig_##name##_,N))(arg1, arg2, arg3); \
} \
static std::vector<SlotInfo*> list_signals_(mo::_counter_<N> dummy) \
{ \
    static SlotInfo info{TypeInfo(typeid(void(ARG1, ARG2, ARG3))), std::string(#name)}; \
    auto signal_info = list_signals_(--dummy); \
    signal_info.push_back(&info); \
    return signal_info; \
}\

#define SIGNAL_5(name, ARG1, ARG2, ARG3, ARG4, N) \
mo::TypedSignal<void(ARG1, ARG2, ARG3, ARG4)>* COMBINE(_sig_##name##_,N) = nullptr; \
inline void sig_##name(ARG1 arg1, ARG2 arg2, ARG3 arg3, ARG4 arg4)\
{\
	if(!_sig_manager) _sig_manager = manager::get_instance(); \
	if(COMBINE(_sig_##name##_,N) == nullptr)\
	{ \
        COMBINE(_sig_##name##_,N) = _sig_manager->GetSignal<void(ARG1, ARG2, ARG3, ARG4)>(#name); \
	} \
	(*COMBINE(_sig_##name##_,N))(arg1, arg2, arg3, arg4); \
} \
static std::vector<mo::SignalInfo*> list_signals_(mo::_counter_<N> dummy) \
{ \
    static mo::SignalInfo info{TypeInfo(typeid(void(ARG1,ARG2,ARG3,ARG4)), std::string(#name)}; \
    auto signal_info = list_signals_(--dummy); \
    signal_info.push_back(&info); \
    return signal_info; \
}\

#define SIGNAL_6(name, ARG1, ARG2, ARG3, ARG4, ARG5, N) \
mo::TypedSignal<void(ARG1, ARG2, ARG3, ARG4, ARG5)>* COMBINE(_sig_##name##_,N) = nullptr; \
inline void sig_##name(ARG1 arg1, ARG2 arg2, ARG3 arg3, ARG4 arg4, ARG5 arg5)\
{\
	if(!_sig_manager) _sig_manager = manager::get_instance(); \
	if(COMBINE(_sig_##name##_,N) == nullptr)\
	{ \
		COMBINE(_sig_##name##_,N) = _sig_manager->GetSignal<void(ARG1, ARG2, ARG3, ARG4, ARG5)>(#name); \
	} \
	(*COMBINE(_sig_##name##_,N))(arg1, arg2, arg3, arg4, arg5); \
} \
static std::vector<mo::SignalInfo*> list_signals_(mo::_counter_<N> dummy) \
{ \
    static mo::SignalInfo info{TypeInfo(typeid(void(ARG1,ARG2,ARG3,ARG4,ARG5)), std::string(#name)}; \
    auto signal_info = list_signals_(--dummy); \
    signal_info.push_back(&info); \
    return signal_info; \
}\

#define SIGNAL_7(name, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, N) \
mo::TypedSignal<void(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6)>* COMBINE(_sig_##name##_,N) = nullptr; \
inline void sig_##name(ARG1 arg1, ARG2 arg2, ARG3 arg3, ARG4 arg4, ARG5 arg5, ARG6 arg6)\
{\
	if(!_sig_manager) _sig_manager = manager::get_instance(); \
	if(COMBINE(_sig_##name##_,N) == nullptr)\
	{ \
        COMBINE(_sig_##name##_,N) = _sig_manager->GetSignal<void(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6)>(#name); \
	} \
	(*COMBINE(_sig_##name##_,N))(arg1, arg2, arg3, arg4, arg5, arg6); \
} \
static std::vector<mo::SignalInfo*> list_signals_(mo::_counter_<N> dummy) \
{ \
    static mo::SignalInfo info{TypeInfo(typeid(void(ARG1,ARG2,ARG3,ARG4,ARG5,ARG6)), std::string(#name)}; \
    auto signal_info = list_signals_(--dummy); \
    signal_info.push_back(&info); \
    return signal_info; \
}\

#define SIGNAL_8(name, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, N) \
mo::TypedSignal<void(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7)>* COMBINE(_sig_##name##_,N) = nullptr; \
inline void sig_##name(ARG1 arg1, ARG2 arg2, ARG3 arg3, ARG4 arg4, ARG5 arg5, ARG6 arg6, ARG7 arg7)\
{\
	if(!_sig_manager) _sig_manager = manager::get_instance(); \
	if(COMBINE(_sig_##name##_,N) == nullptr)\
	{ \
        COMBINE(_sig_##name##_,N) = _sig_manager->GetSignal<void(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7)>(#name); \
	} \
	(*COMBINE(_sig_##name##_,N))(arg1, arg2, arg3, arg4, arg5, arg6, arg7); \
} \
static std::vector<mo::SignalInfo*> list_signals_(mo::_counter_<N> dummy) \
{ \
    static mo::SignalInfo info{TypeInfo(typeid(void(ARG1,ARG2,ARG3,ARG4,ARG5,ARG6,ARG7)), std::string(#name)}; \
    auto signal_info = list_signals_(--dummy); \
    signal_info.push_back(&info); \
    return signal_info; \
}\

#define SIGNAL_9(name, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, N) \
mo::TypedSignal<void(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8)>* COMBINE(_sig_##name##_,N) = nullptr; \
inline void sig_##name(ARG1 arg1, ARG2 arg2, ARG3 arg3, ARG4 arg4, ARG5 arg5, ARG6 arg6, ARG7 arg7, ARG8 arg8)\
{\
	if(!_sig_manager) _sig_manager = manager::get_instance(); \
	if(COMBINE(_sig_##name##_,N) == nullptr)\
	{ \
        COMBINE(_sig_##name##_,N) = _sig_manager->GetSignal<void(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8)>(#name); \
	} \
	(*COMBINE(_sig_##name##_,N))(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); \
} \
static std::vector<mo::SignalInfo*> list_signals_(mo::_counter_<N> dummy) \
{ \
    static mo::SignalInfo info{TypeInfo(typeid(void(ARG1,ARG2,ARG3,ARG4,ARG5,ARG6,ARG7,ARG8)), std::string(#name)}; \
    auto signal_info = list_signals_(--dummy); \
    signal_info.push_back(&info); \
    return signal_info; \
}\

#define SIGNAL_10(name, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9, N) \
mo::TypedSignal<void(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9)>* COMBINE(_sig_##name##_,N) = nullptr; \
inline void sig_##name(ARG1 arg1, ARG2 arg2, ARG3 arg3, ARG4 arg4, ARG5 arg5, ARG6 arg6, ARG7 arg7, ARG8 arg8, ARG9& arg9)\
{\
	if(!_sig_manager) _sig_manager = manager::get_instance(); \
	if(COMBINE(_sig_##name##_,N) == nullptr)\
	{ \
        COMBINE(_sig_##name##_,N) = _sig_manager->GetSignal<void(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9)>(#name); \
	} \
	(*COMBINE(_sig_##name##_,N))(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); \
} \
static std::vector<mo::SignalInfo*> list_signals_(mo::_counter_<N> dummy) \
{ \
    static mo::SignalInfo info{TypeInfo(typeid(void(ARG1,ARG2,ARG3,ARG4,ARG5,ARG6,ARG7,ARG8,ARG9)), std::string(#name)}; \
    auto signal_info = list_signals_(--dummy); \
    signal_info.push_back(&info); \
    return signal_info; \
}\

#define DESCRIBE_SIGNAL_(NAME, DESCRIPTION, N) \
std::vector<slot_info> list_signals_(mo::_counter_<N> dummy) \
{ \
    auto signal_info = list_signals_(mo::_counter_<N-1>()); \
    for(auto& info : signal_info) \
    { \
        if(info.name == #NAME) \
        { \
            info.description = DESCRIPTION; \
        } \
    } \
    return signal_info; \
}\

#define DESCRIBE_SIGNAL(NAME, DESCRIPTION)