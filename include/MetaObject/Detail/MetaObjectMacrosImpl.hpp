#pragma once
#include "ObjectInterfacePerModule.h"
#include "MetaObject/MetaObjectInfo.hpp"

#define CONNECT_START(N_) \
template<int N> bool connect_(std::string name, mo::ISignal* signal, mo::_counter_<N> dummy) \
{ \
	return connect_(name, signal, --dummy); \
} \
template<int N> int connect_(std::string name, mo::SignalManager* manager, mo::_counter_<N> dummy) \
{ \
	return connect_(name, manager, --dummy); \
} \
template<int N> int connect_(mo::SignalManager* manager, mo::_counter_<N> dummy) \
{ \
	return connect_(manager, --dummy); \
} \
bool connect_(std::string name, mo::ISignal* signal, mo::_counter_<N_> dummy) \
{ \
	return false; \
} \
int connect_(std::string name, mo::SignalManager* manager, mo::_counter_<N_> dummy) \
{ \
	return 0; \
} \
int connect_(mo::SignalManager* manager, mo::_counter_<N_> dummy) \
{ \
	return 0; \
}

#define CONNECT_END(N) \
bool ConnectByName(const std::string& name, mo::ISignal* signal) \
{  \
	if(call_parent_connect<THIS_CLASS>(name, signal, nullptr)) \
		return true; \
    return connect_(name, signal, mo::_counter_<N-1>()); \
} \
int ConnectByName(const std::string& name, SignalManager* manager) \
{ \
	int parent_count = call_parent_connect_by_name<THIS_CLASS>(name, manager, nullptr); \
	return connect_(name, manager, mo::_counter_<N-1>()) + parent_count; \
} \
int Connect(SignalManager* manager) \
{ \
    return connect_(manager, mo::_counter_<N-1>()); \
}\
int ConnectAll(mo::SignalManager* mgr) \
{ \
    return 0; \
} \
int ConnectByName(const std::string& name, mo::IMetaObject* obj) \
{ \
    return 0; \
} \

#define DISCONNECT_START(N_) \
template<int N> int disconnect_(mo::SignalManager* manager_, mo::_counter_<N> dummy) \
{ \
	return disconnect_(manager_, --dummy); \
} \
template<int N> int disconnect_by_name(std::string name, mo::SignalManager* manager_, mo::_counter_<N> dummy) \
{ \
	return disconnect_by_name(name, manager_, --dummy); \
} \
int disconnect_(mo::SignalManager* manager_, mo::_counter_<N_> dummy)\
{ \
	return 0; \
} \
int disconnect_by_name(std::string name, mo::SignalManager* manager_, mo::_counter_<N_> dummy) \
{ \
	return 0; \
}

#define DISCONNECT_END(N) \
int disconnect(std::string name, SignalManager* manager_) \
{ \
	return disconnect_by_name(name, manager_, mo::_counter_<N-1>()); \
} \
int disconnect(SignalManager* manager_) \
{ \
	return disconnect_(manager_, mo::_counter_<N-1>()); \
}


#define SIGNAL_INFO_START(N_) \
template<int N> static void list_signal_info_(std::vector<mo::SignalInfo*>& info, mo::_counter_<N> dummy) \
{ \
    return list_signal_info_(info, --dummy); \
} \
static void list_signal_info_(std::vector<mo::SignalInfo*>& info, mo::_counter_<N_> dummy) \
{ \
} 

#define SIGNAL_INFO_END(N) \
static std::vector<mo::SignalInfo*> GetSignalInfoStatic() \
{ \
    std::vector<mo::SignalInfo*> info; \
    list_signal_info_(info, mo::_counter_<N-1>()); \
    return info; \
} \
std::vector<mo::SignalInfo*> GetSignalInfo() const \
{ \
    return GetSignalInfoStatic(); \
}

#define SIGNALS_START(N_) \
template<int N> void init_signals_(SignalManager* mgr, mo::_counter_<N> dummy) \
{ \
    init_signals_(mgr, --dummy); \
} \
void init_signals_(SignalManager* mgr, mo::_counter_<N_> dummy) \
{ \
}


#define SIGNALS_END(N_)


#define SLOT_INFO_START(N_) \
template<int N> static void list_slots_(std::vector<mo::SlotInfo*>& info, mo::_counter_<N> dummy) \
{ \
    return list_slots_(info, --dummy); \
} \
static void list_slots_(std::vector<mo::SlotInfo*>& info, mo::_counter_<N_> dummy) \
{ \
}

#define SLOT_INFO_END(N) \
static std::vector<mo::SlotInfo*> GetSlotInfoStatic() \
{ \
    std::vector<mo::SlotInfo*> info; \
    list_slots_(info, mo::_counter_<N-1>()); \
    return info; \
} \
std::vector<mo::SlotInfo*> GetSlotInfo() const \
{ \
    return GetSlotInfoStatic(); \
}

#define CALLBACK_INFO_START(N_) \
template<int N> static void list_callbacks_(std::vector<mo::CallbackInfo*>& info, mo::_counter_<N> dummy) \
{ \
    return list_callbacks_(info, --dummy); \
} \
static void list_callbacks_(std::vector<mo::CallbackInfo*>& info, mo::_counter_<N_> dummy) \
{ \
}

#define CALLBACK_INFO_END(N) \
static std::vector<mo::CallbackInfo*> GetCallbackInfoStatic()\
{ \
    std::vector<mo::CallbackInfo*> info; \
    list_callbacks_(info, mo::_counter_<N-1>()); \
    return info; \
} \
std::vector<mo::CallbackInfo*> GetCallbackInfo() const \
{ \
    return GetCallbackInfoStatic(); \
}

#define CALLBACK_START(N_) \
template<int N> int connect_callbacks_(mo::IMetaObject* obj, bool queue, mo::_counter_<N> dummy) \
{ \
    return connect_callbacks_(obj, queue, --dummy); \
} \
int connect_callbacks_(mo::IMetaObject* obj, bool queue, mo::_counter_<N_> dummy) \
{ \
    return 0; \
} \
template<int N> int connect_callbacks_(mo::IMetaObject* obj)



#define CALLBACK_END(N_) \
int ConnectCallbacks(mo::IMetaObject* obj, bool queue) \
{ \
    return connect_callbacks_(obj, queue, mo::_counter_<N_-1>()); \
}


#define PARAMETER_INFO_START(N_) \
template<int N> static void list_parameter_info_(std::vector<mo::ParameterInfo*>& info, mo::_counter_<N> dummy) \
{ \
    list_parameter_info_(info, --dummy); \
} \
static void list_parameter_info_(std::vector<mo::ParameterInfo*>& info, mo::_counter_<N_> dummy) \
{ \
}

#define PARAMETER_INFO_END(N) \
static std::vector<mo::ParameterInfo*> GetParameterInfoStatic() \
{ \
    std::vector<mo::ParameterInfo*> info; \
    list_parameter_info_(info, mo::_counter_<N-1>()); \
    return info; \
} \
std::vector<mo::ParameterInfo*> GetParameterInfo() const \
{ \
    return GetParameterInfoStatic(); \
}

#define SLOT_START(N_) \
template<int N> void bind_slots_(mo::_counter_<N> dummy) \
{ \
    bind_slots_(--dummy); \
} \
void bind_slots_(mo::_counter_<N_> dummy)  \
{  \
}

#define SLOT_END(N_) \
void BindSlots() \
{ \
    bind_slots_(mo::_counter_<N_-1>()); \
}




#define MO_BEGIN_1(CLASS_NAME, N_) \
typedef CLASS_NAME THIS_CLASS;      \
CONNECT_START(N_) \
DISCONNECT_START(N_) \
CALLBACK_INFO_START(N_) \
SIGNAL_INFO_START(N_) \
SIGNALS_START(N_) \
SLOT_INFO_START(N_) \
PARAMETER_INFO_START(N_) \
SLOT_START(N_)

#define MO_BEGIN_2(CLASS_NAME, PARENT, N_) \
typedef CLASS_NAME THIS_CLASS; \
typedef PARENT PARENT_CLASS;  \
CONNECT_START(N_) \
DISCONNECT_START(N_) \
CALLBACK_INFO_START(N_) \
SIGNAL_INFO_START(N_) \
SIGNALS_START(N_) \
SLOT_INFO_START(N_) \
PARAMETER_INFO_START(N_) \
SLOT_START(N_)

#define MO_END_(N) \
template<typename T> int call_parent_setup(mo::SignalManager* manager, typename std::enable_if<mo::has_parent<T>::value, void>::type* = nullptr) \
{ \
	LOG(trace) << typeid(T::PARENT_CLASS).name(); \
    return T::PARENT_CLASS::SetupSignals(manager); \
} \
template<typename T> int call_parent_setup(mo::SignalManager* manager, typename std::enable_if<!mo::has_parent<T>::value, void>::type* = nullptr) \
{ \
	return 0; \
} \
template<typename T> int call_parent_connect_by_name(const std::string& name, mo::SignalManager* manager, typename std::enable_if<mo::has_parent<T>::value, void>::type* = nullptr) \
{ \
	LOG(trace) << typeid(T::PARENT_CLASS).name(); \
    return T::PARENT_CLASS::connect_by_name(name, manager); \
} \
template<typename T> int call_parent_connect_by_name(const std::string& name, mo::SignalManager* manager, typename std::enable_if<!mo::has_parent<T>::value, void>::type* = nullptr) \
{ \
	return 0; \
} \
template<typename T> bool call_parent_connect(const std::string& name, mo::ISignal* signal, typename std::enable_if<mo::has_parent<T>::value, void>::type* = nullptr) \
{ \
	LOG(trace) << typeid(T::PARENT_CLASS).name(); \
    return T::PARENT_CLASS::connect(name, signal); \
} \
template<typename T> bool call_parent_connect(const std::string& name, mo::ISignal* signal, typename std::enable_if<!mo::has_parent<T>::value, void>::type* = nullptr) \
{ \
	return false; \
} \
virtual int SetupSignals(mo::SignalManager* manager) \
{ \
    int parent_signal_count = call_parent_setup<THIS_CLASS>(manager, nullptr); \
    _sig_manager = manager; \
    bind_slots_(mo::_counter_<N-1>()); \
    init_signals_(manager, mo::_counter_<N-1>()); \
    return Connect(manager) + parent_signal_count; \
} \
CONNECT_END(N) \
DISCONNECT_END(N) \
CALLBACK_INFO_END(N) \
SIGNAL_INFO_END(N) \
SLOT_INFO_END(N) \
PARAMETER_INFO_END(N) \
SIGNALS_END(N) \
SLOT_END(N)

#define MO_REGISTER_OBJECT(TYPE) \
    static mo::MetaObjectInfo<TActual<TYPE>, __COUNTER__> TYPE##_info; \
    REGISTERCLASS(TYPE, &TYPE##_info);