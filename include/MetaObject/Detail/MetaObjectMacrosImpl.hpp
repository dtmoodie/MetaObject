#pragma once
#ifndef __CUDACC__
#include "MetaObject/MetaObjectInfo.hpp"
#include "MetaObject/MetaObjectPolicy.hpp"
#include "MetaObject/MetaObjectFactory.hpp"
#include "MetaObject/Detail/Counter.hpp"
#include <boost/preprocessor.hpp>
#include <RuntimeObjectSystem/shared_ptr.hpp>
#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"
#include <type_traits>
#include <vector>
#include <string>

struct ISimpleSerializer;
namespace mo {
struct SignalInfo;
struct SlotInfo;
struct ParamInfo;
}

// ---------------- SIGNAL_INFO ------------
#define SIGNAL_INFO_START(N_) \
template<int N> static void list_signal_info_(std::vector<mo::SignalInfo*>& info, mo::_counter_<N> dummy){ \
    return list_signal_info_(info, --dummy); \
} \
static void list_signal_info_(std::vector<mo::SignalInfo*>& info, mo::_counter_<N_> dummy){ \
    (void)info; \
    (void)dummy; \
}

#define SIGNAL_INFO_END(N) \
static void getSignalInfoStatic(std::vector<mo::SignalInfo*> & info){ \
    list_signal_info_(info, mo::_counter_<N-1>()); \
    _list_parent_signals(info); \
    std::sort(info.begin(), info.end()); \
    info.erase(std::unique(info.begin(), info.end()), info.end()); \
} \
static std::vector<mo::SignalInfo*> getSignalInfoStatic(){ \
    std::vector<mo::SignalInfo*> info; \
    getSignalInfoStatic(info); \
    return info; \
} \
void getSignalInfo(std::vector<mo::SignalInfo*> & info) const{ \
    getSignalInfoStatic(info); \
}

// ---------------- SIGNALS ------------
#define SIGNALS_START(N_) \
template<int N> int init_signals_(bool firstInit, mo::_counter_<N> dummy){ \
    return init_signals_(firstInit, --dummy); \
} \
int init_signals_(bool firstInit, mo::_counter_<N_> dummy){ \
    (void)dummy; \
    (void)firstInit; \
    return 0; \
}

#define SIGNALS_END(N_) \
virtual int InitSignals(bool firstInit){ \
    int count = _init_parent_signals(firstInit); \
    return init_signals_(firstInit, mo::_counter_<N_-1>()) + count; \
}

// ---------------- SLOT INFO ------------
#define SLOT_INFO_START(N_) \
template<int N> static void list_slots_(std::vector<mo::SlotInfo*>& info, mo::_counter_<N> dummy){ \
    return list_slots_(info, --dummy); \
} \
static void list_slots_(std::vector<mo::SlotInfo*>& info, mo::_counter_<N_> dummy){ \
    (void)info; \
    (void)dummy; \
}

#define SLOT_INFO_END(N) \
static void getSlotInfoStatic(std::vector<mo::SlotInfo*> & info){ \
    list_slots_(info, mo::_counter_<N-1>()); \
    _list_parent_slots(info); \
    std::sort(info.begin(), info.end()); \
    info.erase(std::unique(info.begin(), info.end()), info.end()); \
} \
static std::vector<mo::SlotInfo*> getSlotInfoStatic(){ \
    std::vector<mo::SlotInfo*> info; \
    getSlotInfoStatic(info); \
    return info; \
} \
void getSlotInfo(std::vector<mo::SlotInfo*>& info) const{ \
    getSlotInfoStatic(info); \
}


// ---------------- ParamS INFO ------------

#define PARAM_INFO_START(N_) \
template<int N> static void list_Param_info_(std::vector<mo::ParamInfo*>& info, mo::_counter_<N> dummy){ \
    list_Param_info_(info, --dummy); \
} \
static void list_Param_info_(std::vector<mo::ParamInfo*>& info, mo::_counter_<N_> dummy){ \
    (void)info; \
    (void)dummy; \
}


#define PARAM_INFO_END(N) \
static void getParamInfoStatic(std::vector<mo::ParamInfo*>& info){ \
    list_Param_info_(info, mo::_counter_<N-1>()); \
    _list_parent_Param_info(info); \
    std::sort(info.begin(), info.end()); \
    info.erase(std::unique(info.begin(), info.end()), info.end()); \
} \
static std::vector<mo::ParamInfo*> getParamInfoStatic(){ \
    std::vector<mo::ParamInfo*> info; \
    getParamInfoStatic(info); \
    return info; \
} \
void getParamInfo(std::vector<mo::ParamInfo*>& info) const{ \
    getParamInfoStatic(info); \
}

// ---------------- ParamS ------------
#define PARAM_START(N_) \
template<int N> void init_Params_(bool firstInit, mo::_counter_<N> dummy){ \
    init_Params_(firstInit, --dummy); \
} \
void init_Params_(bool firstInit, mo::_counter_<N_> dummy){ \
    (void)firstInit; \
    (void)dummy; \
} \
template<int N> void _serialize_Params(ISimpleSerializer* pSerializer, mo::_counter_<N> dummy){ \
    _serialize_Params(pSerializer, --dummy); \
} \
void _serialize_Params(ISimpleSerializer* pSerializer, mo::_counter_<N_> dummy){ \
    (void)dummy; \
    (void)pSerializer; \
} \
template<class T, int N> void _load_Params(T& ar, mo::_counter_<N> dummy){ \
    _load_Params<T>(ar, --dummy); \
} \
template<class T, int N> void _save_Params(T& ar, mo::_counter_<N> dummy) const{ \
    _save_Params<T>(ar, --dummy); \
} \
template<class T> void _load_Params(T& ar, mo::_counter_<N_> dummy){ \
    (void)dummy; \
    (void)ar; \
} \
template<class T> void _save_Params(T& ar, mo::_counter_<N_> dummy) const{ \
    (void)dummy; \
    (void)ar; \
}

#define PARAM_END(N_) \
void initParams(bool firstInit){ \
    init_Params_(firstInit, mo::_counter_<N_ - 1>()); \
    _init_parent_params(firstInit); \
} \
void serializeParams(ISimpleSerializer* pSerializer){ \
    _serialize_Params(pSerializer, mo::_counter_<N_ - 1>()); \
    _serialize_parent_params(pSerializer); \
} \
template<class T> void load(T& ar){ \
    _load_Params<T>(ar, mo::_counter_<N_ -1>()); \
    _load_parent<T>(ar); \
} \
template<class T> void save(T& ar) const{ \
    _save_Params<T>(ar, mo::_counter_<N_ -1>()); \
    _save_parent<T>(ar); \
}


// -------------- SLOTS -------------
#define SLOT_START(N_) \
template<int N> void bind_slots_(bool firstInit, mo::_counter_<N> dummy){ \
    bind_slots_(firstInit, --dummy); \
} \
void bind_slots_(bool firstInit, mo::_counter_<N_> dummy){  \
    (void)dummy; \
    (void)firstInit; \
}

#define SLOT_END(N_) \
void bindSlots(bool firstInit){ \
    _bind_parent_slots(firstInit); \
    bind_slots_(firstInit, mo::_counter_<N_-1>()); \
}

#define HANDLE_PARENT_1(PARENT1) \
void _init_parent_params(bool firstInit){ \
    PARENT1::initParams(firstInit); \
} \
void _serialize_parent_params(ISimpleSerializer* pSerializer){ \
    PARENT1::serializeParams(pSerializer); \
} \
template<class T> void _load_parent(T& ar){ \
    PARENT1::load(ar); \
} \
template<class T> void _save_parent(T& ar) const{ \
   PARENT1::save(ar); \
} \
void _bind_parent_slots(bool firstInit){ \
    PARENT1::bindSlots(firstInit); \
} \
static void _list_parent_Param_info(std::vector<mo::ParamInfo*>& info){ \
    PARENT1::getParamInfoStatic(info); \
} \
static void _list_parent_signals(std::vector<mo::SignalInfo*>& info){ \
    PARENT1::getSignalInfoStatic(info); \
} \
static void _list_parent_slots(std::vector<mo::SlotInfo*>& info){ \
    PARENT1::getSlotInfoStatic(info); \
} \
int _init_parent_signals(bool firstInit){ \
    return PARENT1::InitSignals(firstInit); \
}


#define HANDLE_PARENT_2(PARENT1, PARENT2) \
void _init_parent_params(bool firstInit){ \
    PARENT1::initParams(firstInit); \
    PARENT2::initParams(firstInit); \
} \
void _serialize_parent_params(ISimpleSerializer* pSerializer){ \
    PARENT1::serializeParams(pSerializer); \
    PARENT2::serializeParams(pSerializer); \
} \
template<class T> void _load_parent(T& ar){ \
    PARENT1::load(ar); \
    PARENT2::load(ar); \
} \
template<class T> void _save_parent(T& ar) const{ \
   PARENT1::save(ar); \
   PARENT2::save(ar); \
} \
void _bind_parent_slots(bool firstInit){ \
    PARENT1::bindSlots(firstInit); \
    PARENT2::bindSlots(firstInit); \
} \
static void _list_parent_Param_info(std::vector<mo::ParamInfo*>& info){ \
    PARENT1::getParamInfoStatic(info); \
    PARENT2::getParamInfoStatic(info); \
} \
static void _list_parent_signals(std::vector<mo::SignalInfo*>& info){ \
    PARENT1::getSignalInfoStatic(info); \
    PARENT2::getSignalInfoStatic(info); \
} \
static void _list_parent_slots(std::vector<mo::SlotInfo*>& info){ \
    PARENT1::getSlotInfoStatic(info); \
    PARENT2::getSlotInfoStatic(info); \
} \
int _init_parent_signals(bool firstInit){ \
    return PARENT1::InitSignals(firstInit) + PARENT2::InitSignals(firstInit);; \
} \

#ifdef _MSC_VER
#define HANDLE_PARENT(...)  BOOST_PP_CAT(BOOST_PP_OVERLOAD(HANDLE_PARENT_, __VA_ARGS__)(__VA_ARGS__), BOOST_PP_EMPTY())
#else
#define HANDLE_PARENT(...)  BOOST_PP_OVERLOAD(HANDLE_PARENT_, __VA_ARGS__)(__VA_ARGS__)
#endif

#define HANDLE_NO_PARENT \
void _init_parent_params(bool firstInit){ (void)firstInit; } \
void _serialize_parent_params(ISimpleSerializer* pSerializer) { (void)pSerializer; } \
template<class T> void _load_parent(T& ar) { (void)ar; } \
template<class T> void _save_parent(T& ar) const { (void)ar;} \
void _bind_parent_slots(bool firstInit) { (void)firstInit;} \
static void _list_parent_Param_info(std::vector<mo::ParamInfo*>& info) { (void)info;} \
static void _list_parent_signals(std::vector<mo::SignalInfo*>& info) { (void)info;} \
static void _list_parent_slots(std::vector<mo::SlotInfo*>& info) { (void)info;} \
int _init_parent_signals(bool firstInit) {(void)firstInit; return 0;}

#define MO_BEGIN_1(CLASS_NAME, N_) \
typedef CLASS_NAME THIS_CLASS;      \
HANDLE_NO_PARENT; \
SIGNAL_INFO_START(N_) \
SIGNALS_START(N_) \
SLOT_INFO_START(N_) \
PARAM_INFO_START(N_) \
SLOT_START(N_) \
PARAM_START(N_) \
static rcc::shared_ptr<CLASS_NAME> Create();

#define MO_DERIVE_(N_, CLASS_NAME, ...) \
typedef CLASS_NAME THIS_CLASS; \
HANDLE_PARENT(__VA_ARGS__); \
SIGNAL_INFO_START(N_) \
SIGNALS_START(N_) \
SLOT_INFO_START(N_) \
Param_INFO_START(N_) \
SLOT_START(N_)\
Param_START(N_) \
static rcc::shared_ptr<CLASS_NAME> Create();

#define MO_END_(N) \
SIGNAL_INFO_END(N) \
SLOT_INFO_END(N) \
PARAM_INFO_END(N) \
SIGNALS_END(N) \
SLOT_END(N) \
PARAM_END(N)

#define MO_ABSTRACT_(N_, CLASS_NAME, ...) \
typedef CLASS_NAME THIS_CLASS; \
HANDLE_PARENT(__VA_ARGS__); \
SIGNAL_INFO_START(N_) \
SIGNALS_START(N_) \
SLOT_INFO_START(N_) \
PARAM_INFO_START(N_) \
SLOT_START(N_)\
PARAM_START(N_)

#define MO_REGISTER_OBJECT(TYPE) \
    static ::mo::MetaObjectInfo<TActual<TYPE>> TYPE##_info; \
    static ::mo::MetaObjectPolicy<TActual<TYPE>, __COUNTER__, void> TYPE##_policy; \
    ::rcc::shared_ptr<TYPE> TYPE::Create() { \
        auto obj = ::mo::MetaObjectFactory::Instance()->Create(#TYPE); \
        return ::rcc::shared_ptr<TYPE>(obj); \
    } \
    REGISTERCLASS(TYPE, &TYPE##_info);

#define MO_REGISTER_CLASS(TYPE) MO_REGISTER_OBJECT(TYPE)

#else // __CUDACC__
#define MO_REGISTER_OBJECT(TYPE)
#define MO_REGISTER_CLASS(TYPE)
#define MO_BEGIN_1(CLASS, N)
#define MO_BEGIN_2(CLASS, PARENT, N)
#define MO_END_(N)
#endif // __CUDACC__
