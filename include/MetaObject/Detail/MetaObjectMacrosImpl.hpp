#pragma once

#define CONNECT_BLOCK(N_) \
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

#define DISCONNECT_BLOCK(N_) \
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

#define INFO_BLOCK(N_) \
template<int N> static std::vector<mo::SignalInfo*> list_signals_(mo::_counter_<N> dummy) \
{ \
    return list_signals_(--dummy); \
} \
template<int N> static std::vector<mo::SlotInfo*> list_slots_(mo::_counter_<N> dummy) \
{ \
    return list_slots_(--dummy); \
} \
static std::vector<mo::SignalInfo*> list_signals_(mo::_counter_<N_> dummy) \
{ \
    return std::vector<mo::SignalInfo*>(); \
} \
static std::vector<mo::SlotInfo*> list_slots_(mo::_counter_<N_> dummy) \
{ \
    return std::vector<mo::SlotInfo*>(); \
}

#define MO_BEGIN_1(CLASS_NAME, N_) \
typedef CLASS_NAME THIS_CLASS;      \
CONNECT_BLOCK(N_) \
DISCONNECT_BLOCK(N_) \
INFO_BLOCK(N_)


#define MO_BEGIN_2(CLASS_NAME, PARENT, N_) \
typedef CLASS_NAME THIS_CLASS; \
typedef PARENT PARENT_CLASS;  \
CONNECT_BLOCK(N_) \
DISCONNECT_BLOCK(N_) \
INFO_BLOCK(N_)

#define MO_END_(N) \
template<typename T> int call_parent_setup(mo::SignalManager* manager, typename std::enable_if<mo::has_parent<T>::value, void>::type* = nullptr) \
{ \
	LOG(trace) << typeid(T::PARENT_CLASS).name(); \
    return T::PARENT_CLASS::setup_signals(manager); \
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
virtual int setup_signals(mo::SignalManager* manager) \
{ \
    int parent_signal_count = call_parent_setup<THIS_CLASS>(manager, nullptr); \
    _sig_manager = manager; \
    return connect(manager) + parent_signal_count; \
} \
bool Connect(std::string name, mo::ISignal* signal) \
{  \
	if(call_parent_connect<THIS_CLASS>(name, signal, nullptr)) \
		return true; \
    return connect_(name, signal, mo::_counter_<N-1>()); \
} \
int connect_by_name(const std::string& name, SignalManager* manager) \
{ \
	int parent_count = call_parent_connect_by_name<THIS_CLASS>(name, manager, nullptr); \
	return connect_(name, manager, mo::_counter_<N-1>()) + parent_count; \
} \
int connect(SignalManager* manager) \
{ \
    return connect_(manager, mo::_counter_<N-1>()); \
}\
int disconnect(std::string name, SignalManager* manager_) \
{ \
	return disconnect_by_name(name, manager_, mo::_counter_<N-1>()); \
} \
int disconnect(SignalManager* manager_) \
{ \
	return disconnect_(manager_, mo::_counter_<N-1>()); \
} \
static std::vector<mo::SignalInfo*> GetSignalInfoStatic() \
{ \
    return list_signals_(mo::_counter_<N-1>()); \
} \
static std::vector<mo::SlotInfo*> GetSlotInfoStatic() \
{ \
    return list_slots_(mo::_counter_<N-1>()); \
} \
std::vector<mo::SignalInfo*> GetSignalInfo() \
{ \
    return GetSignalInfoStatic(); \
} \
std::vector<mo::SlotInfo*> GetSlotInfo() \
{ \
    return GetSlotInfoStatic(); \
}