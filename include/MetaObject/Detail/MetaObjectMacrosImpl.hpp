#pragma once
#include "ObjectInterfacePerModule.h"
#include "MetaObject/MetaObjectInfo.hpp"


// ---------------- SIGNAL_INFO ------------
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

// ---------------- SIGNALS ------------
#define SIGNALS_START(N_) \
template<int N> int init_signals_(bool firstInit, mo::_counter_<N> dummy) \
{ \
    return init_signals_(firstInit, --dummy); \
} \
int init_signals_(bool firstInit, mo::_counter_<N_> dummy) \
{ \
    return 0; \
}


#define SIGNALS_END(N_) \
virtual int InitSignals(bool firstInit) \
{ \
	return init_signals_(firstInit, mo::_counter_<N_-1>()); \
}

// ---------------- SLOT INFO ------------
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
// ---------------- CALLBACK INFO ------------
#define CALLBACK_INFO_START(N_) \
template<int N> static void list_callbacks_(std::vector<mo::CallbackInfo*>& info, mo::_counter_<N> dummy) \
{ \
    return list_callbacks_(info, --dummy); \
} \
static void list_callbacks_(std::vector<mo::CallbackInfo*>& info, mo::_counter_<N_> dummy) \
{ \
} \


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

// ---------------- CALLBACKS ------------
#define CALLBACK_START(N_) \
template<int N> void add_callbacks_(mo::_counter_<N> dummy) \
{ \
    add_callbacks_(--dummy); \
} \
void add_callbacks_(mo::_counter_<N_> dummy) \
{ \
}

#define CALLBACK_END(N_)


// ---------------- PARAMETERS INFO ------------

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

// ---------------- PARAMETERS ------------
#define PARAMETER_START(N_) \
template<int N> void init_parameters_(bool firstInit, mo::_counter_<N> dummy) \
{ \
    init_parameters_(firstInit, --dummy); \
} \
void init_parameters_(bool firstInit, mo::_counter_<N_> dummy) \
{ \
}

#define PARAMETER_END(N_) \
void InitParameters(bool firstInit) \
{ \
    init_parameters_(firstInit, mo::_counter_<N_ - 1>()); \
}


// -------------- SLOTS -------------
#define SLOT_START(N_) \
template<int N> void bind_slots_(bool firstInit, mo::_counter_<N> dummy) \
{ \
    bind_slots_(firstInit, --dummy); \
} \
void bind_slots_(bool firstInit, mo::_counter_<N_> dummy)  \
{  \
}

#define SLOT_END(N_) \
void BindSlots(bool firstInit) \
{ \
    bind_slots_(firstInit, mo::_counter_<N_-1>()); \
}




#define MO_BEGIN_1(CLASS_NAME, N_) \
typedef CLASS_NAME THIS_CLASS;      \
CALLBACK_INFO_START(N_) \
CALLBACK_START(N_) \
SIGNAL_INFO_START(N_) \
SIGNALS_START(N_) \
SLOT_INFO_START(N_) \
PARAMETER_INFO_START(N_) \
SLOT_START(N_) \
PARAMETER_START(N_) \
static rcc::shared_ptr<CLASS_NAME> Create();

#define MO_BEGIN_2(CLASS_NAME, PARENT, N_) \
typedef CLASS_NAME THIS_CLASS; \
typedef PARENT PARENT_CLASS;  \
CALLBACK_INFO_START(N_) \
CALLBACK_START(N_) \
SIGNAL_INFO_START(N_) \
SIGNALS_START(N_) \
SLOT_INFO_START(N_) \
PARAMETER_INFO_START(N_) \
SLOT_START(N_)\
PARAMETER_START(N_)

#define MO_END_(N) \
template<typename T> int call_parent_setup(mo::RelayManager* manager, typename std::enable_if<mo::has_parent<T>::value, void>::type* = nullptr) \
{ \
	LOG(trace) << typeid(T::PARENT_CLASS).name(); \
    return T::PARENT_CLASS::SetupSignals(manager); \
} \
template<typename T> int call_parent_setup(mo::RelayManager* manager, typename std::enable_if<!mo::has_parent<T>::value, void>::type* = nullptr) \
{ \
	return 0; \
} \
template<typename T> int call_parent_connect_by_name(const std::string& name, mo::RelayManager* manager, typename std::enable_if<mo::has_parent<T>::value, void>::type* = nullptr) \
{ \
	LOG(trace) << typeid(T::PARENT_CLASS).name(); \
    return T::PARENT_CLASS::connect_by_name(name, manager); \
} \
template<typename T> int call_parent_connect_by_name(const std::string& name, mo::RelayManager* manager, typename std::enable_if<!mo::has_parent<T>::value, void>::type* = nullptr) \
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
virtual int SetupSignals(mo::RelayManager* manager) \
{ \
    int parent_signal_count = call_parent_setup<THIS_CLASS>(manager, nullptr); \
    _sig_manager = manager; \
    return parent_signal_count; \
} \
CALLBACK_INFO_END(N) \
CALLBACK_END(N) \
SIGNAL_INFO_END(N) \
SLOT_INFO_END(N) \
PARAMETER_INFO_END(N) \
SIGNALS_END(N) \
SLOT_END(N) \
PARAMETER_END(N)


#define MO_REGISTER_OBJECT(TYPE) \
    static mo::MetaObjectInfo<TActual<TYPE>, __COUNTER__> TYPE##_info; \
    rcc::shared_ptr<TYPE> TYPE::Create() \
    { \
        auto obj = IRuntimeObjectSystem::Instance()->GetObjectFactorySystem()->GetConstructor(#TYPE)->Construct(); \
        obj->Init(true); \
        return rcc::shared_ptr<TYPE>(obj); \
    } \
    REGISTERCLASS(TYPE, &TYPE##_info);