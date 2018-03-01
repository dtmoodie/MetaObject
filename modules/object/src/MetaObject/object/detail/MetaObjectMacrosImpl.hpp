#pragma once
#ifndef __CUDACC__
#include "MetaObject/core/detail/Counter.hpp"
#include "MetaObject/object/MetaObjectFactory.hpp"
#include "MetaObject/object/MetaObjectInfo.hpp"
#include "MetaObject/object/MetaObjectPolicy.hpp"
#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"
#include <RuntimeObjectSystem/shared_ptr.hpp>
#include <boost/preprocessor.hpp>
#include <string>
#include <type_traits>
#include <vector>

struct ISimpleSerializer;
namespace mo
{
    struct SignalInfo;
    struct SlotInfo;
    struct ParamInfo;
}

// ---------------- SIGNAL_INFO ------------
#define SIGNAL_INFO_START(N_)                                                                                          \
    template <int N>                                                                                                   \
    static void _list_signal_info(std::vector<mo::SignalInfo*>& info, mo::_counter_<N> dummy)                          \
    {                                                                                                                  \
        return _list_signal_info(info, --dummy);                                                                       \
    }                                                                                                                  \
    static void _list_signal_info(std::vector<mo::SignalInfo*>& info, mo::_counter_<N_> dummy)                         \
    {                                                                                                                  \
        (void)info;                                                                                                    \
        (void)dummy;                                                                                                   \
    }

#define SIGNAL_INFO_END(N)                                                                                             \
    static void getSignalInfoStatic(std::vector<mo::SignalInfo*>& info)                                                \
    {                                                                                                                  \
        _list_signal_info(info, mo::_counter_<N - 1>());                                                               \
        _list_parent_signals(info);                                                                                    \
        std::sort(info.begin(), info.end());                                                                           \
        info.erase(std::unique(info.begin(), info.end()), info.end());                                                 \
    }                                                                                                                  \
    static std::vector<mo::SignalInfo*> getSignalInfoStatic()                                                          \
    {                                                                                                                  \
        std::vector<mo::SignalInfo*> info;                                                                             \
        getSignalInfoStatic(info);                                                                                     \
        return info;                                                                                                   \
    }                                                                                                                  \
    virtual void getSignalInfo(std::vector<mo::SignalInfo*>& info) const override { getSignalInfoStatic(info); }

// ---------------- SIGNALS ------------
#define SIGNALS_START(N_)                                                                                              \
    template <int N>                                                                                                   \
    int _init_signals(bool firstInit, mo::_counter_<N> dummy)                                                          \
    {                                                                                                                  \
        return _init_signals(firstInit, --dummy);                                                                      \
    }                                                                                                                  \
    int _init_signals(bool firstInit, mo::_counter_<N_> dummy)                                                         \
    {                                                                                                                  \
        (void)dummy;                                                                                                   \
        (void)firstInit;                                                                                               \
        return 0;                                                                                                      \
    }

#define SIGNALS_END(N_)                                                                                                \
    virtual int initSignals(bool firstInit) override                                                                   \
    {                                                                                                                  \
        int count = _init_parent_signals(firstInit);                                                                   \
        return _init_signals(firstInit, mo::_counter_<N_ - 1>()) + count;                                              \
    }

// ---------------- SLOT INFO ------------
#define SLOT_INFO_START(N_)                                                                                            \
    template <int N>                                                                                                   \
    static void _list_slots(std::vector<mo::SlotInfo*>& info, mo::_counter_<N> dummy)                                  \
    {                                                                                                                  \
        return _list_slots(info, --dummy);                                                                             \
    }                                                                                                                  \
    static void _list_slots(std::vector<mo::SlotInfo*>& info, mo::_counter_<N_> dummy)                                 \
    {                                                                                                                  \
        (void)info;                                                                                                    \
        (void)dummy;                                                                                                   \
    }

#define SLOT_INFO_END(N)                                                                                               \
    static void getSlotInfoStatic(std::vector<mo::SlotInfo*>& info)                                                    \
    {                                                                                                                  \
        _list_slots(info, mo::_counter_<N - 1>());                                                                     \
        _list_parent_slots(info);                                                                                      \
        std::sort(info.begin(), info.end());                                                                           \
        info.erase(std::unique(info.begin(), info.end()), info.end());                                                 \
    }                                                                                                                  \
    static std::vector<mo::SlotInfo*> getSlotInfoStatic()                                                              \
    {                                                                                                                  \
        std::vector<mo::SlotInfo*> info;                                                                               \
        getSlotInfoStatic(info);                                                                                       \
        return info;                                                                                                   \
    }                                                                                                                  \
    virtual void getSlotInfo(std::vector<mo::SlotInfo*>& info) const override { getSlotInfoStatic(info); }

// ---------------- ParamS INFO ------------

#define PARAM_INFO_START(N_)                                                                                           \
    template <int N>                                                                                                   \
    static void _list_param_info(std::vector<mo::ParamInfo*>& info, mo::_counter_<N> dummy)                            \
    {                                                                                                                  \
        _list_param_info(info, --dummy);                                                                               \
    }                                                                                                                  \
    static void _list_param_info(std::vector<mo::ParamInfo*>& info, mo::_counter_<N_> dummy)                           \
    {                                                                                                                  \
        (void)info;                                                                                                    \
        (void)dummy;                                                                                                   \
    }

#define PARAM_INFO_END(N)                                                                                              \
    static void getParamInfoStatic(std::vector<mo::ParamInfo*>& info)                                                  \
    {                                                                                                                  \
        _list_param_info(info, mo::_counter_<N - 1>());                                                                \
        _list_parent_param_info(info);                                                                                 \
        std::sort(info.begin(), info.end());                                                                           \
        info.erase(std::unique(info.begin(), info.end()), info.end());                                                 \
    }                                                                                                                  \
    static std::vector<mo::ParamInfo*> getParamInfoStatic()                                                            \
    {                                                                                                                  \
        std::vector<mo::ParamInfo*> info;                                                                              \
        getParamInfoStatic(info);                                                                                      \
        return info;                                                                                                   \
    }                                                                                                                  \
    virtual void getParamInfo(std::vector<mo::ParamInfo*>& info) const override { getParamInfoStatic(info); }

// ---------------- Params ------------
#define PARAM_START(N_)                                                                                                \
    template <int N>                                                                                                   \
    void _init_params(bool firstInit, mo::_counter_<N> dummy)                                                          \
    {                                                                                                                  \
        _init_params(firstInit, --dummy);                                                                              \
    }                                                                                                                  \
    void _init_params(bool firstInit, mo::_counter_<N_> dummy)                                                         \
    {                                                                                                                  \
        (void)firstInit;                                                                                               \
        (void)dummy;                                                                                                   \
    }                                                                                                                  \
    template <int N>                                                                                                   \
    void _init_outputs(mo::_counter_<N> dummy)                                                                         \
    {                                                                                                                  \
        _init_outputs(--dummy);                                                                                        \
    }                                                                                                                  \
    void _init_outputs(mo::_counter_<N_> dummy) { (void)dummy; }                                                       \
    template <int N>                                                                                                   \
    void _serialize_params(ISimpleSerializer* pSerializer, mo::_counter_<N> dummy)                                     \
    {                                                                                                                  \
        _serialize_params(pSerializer, --dummy);                                                                       \
    }                                                                                                                  \
    void _serialize_params(ISimpleSerializer* pSerializer, mo::_counter_<N_> dummy)                                    \
    {                                                                                                                  \
        (void)dummy;                                                                                                   \
        (void)pSerializer;                                                                                             \
    }                                                                                                                  \
    template <class T, int N>                                                                                          \
    void _load_params(T& ar, mo::_counter_<N> dummy)                                                                   \
    {                                                                                                                  \
        _load_params<T>(ar, --dummy);                                                                                  \
    }                                                                                                                  \
    template <class T, int N>                                                                                          \
    void _save_params(T& ar, mo::_counter_<N> dummy) const                                                             \
    {                                                                                                                  \
        _save_params<T>(ar, --dummy);                                                                                  \
    }                                                                                                                  \
    template <class T>                                                                                                 \
    void _load_params(T& ar, mo::_counter_<N_> dummy)                                                                  \
    {                                                                                                                  \
        (void)dummy;                                                                                                   \
        (void)ar;                                                                                                      \
    }                                                                                                                  \
    template <class T>                                                                                                 \
    void _save_params(T& ar, mo::_counter_<N_> dummy) const                                                            \
    {                                                                                                                  \
        (void)dummy;                                                                                                   \
        (void)ar;                                                                                                      \
    }

#define PARAM_END(N_)                                                                                                  \
    virtual void initParams(bool firstInit) override                                                                   \
    {                                                                                                                  \
        _init_params(firstInit, mo::_counter_<N_ - 1>());                                                              \
        _init_parent_params(firstInit);                                                                                \
    }                                                                                                                  \
    virtual void serializeParams(ISimpleSerializer* pSerializer) override                                              \
    {                                                                                                                  \
        _serialize_params(pSerializer, mo::_counter_<N_ - 1>());                                                       \
        _serialize_parent_params(pSerializer);                                                                         \
    }                                                                                                                  \
    virtual void initOutputs() override                                                                                \
    {                                                                                                                  \
        _init_outputs(mo::_counter_<N_ - 1>());                                                                        \
        _init_parent_outputs();                                                                                        \
    }                                                                                                                  \
    template <class T>                                                                                                 \
    void load(T& ar)                                                                                                   \
    {                                                                                                                  \
        _load_params<T>(ar, mo::_counter_<N_ - 1>());                                                                  \
        _load_parent<T>(ar);                                                                                           \
    }                                                                                                                  \
    template <class T>                                                                                                 \
    void save(T& ar) const                                                                                             \
    {                                                                                                                  \
        _save_params<T>(ar, mo::_counter_<N_ - 1>());                                                                  \
        _save_parent<T>(ar);                                                                                           \
    }

// -------------- SLOTS -------------
#define SLOT_START(N_)                                                                                                 \
    template <int N>                                                                                                   \
    void _bind_slots(bool firstInit, mo::_counter_<N> dummy)                                                           \
    {                                                                                                                  \
        _bind_slots(firstInit, --dummy);                                                                               \
    }                                                                                                                  \
    void _bind_slots(bool firstInit, mo::_counter_<N_> dummy)                                                          \
    {                                                                                                                  \
        (void)dummy;                                                                                                   \
        (void)firstInit;                                                                                               \
    }

#define SLOT_END(N_)                                                                                                   \
    virtual void bindSlots(bool firstInit) override                                                                    \
    {                                                                                                                  \
        _bind_parent_slots(firstInit);                                                                                 \
        _bind_slots(firstInit, mo::_counter_<N_ - 1>());                                                               \
    }

#define HANDLE_PARENT_1(PARENT1)                                                                                       \
    void _init_parent_params(bool firstInit) { PARENT1::initParams(firstInit); }                                       \
    void _init_parent_outputs() { PARENT1::initOutputs(); }                                                            \
    void _serialize_parent_params(ISimpleSerializer* pSerializer) { PARENT1::serializeParams(pSerializer); }           \
    template <class T>                                                                                                 \
    void _load_parent(T& ar)                                                                                           \
    {                                                                                                                  \
        PARENT1::load(ar);                                                                                             \
    }                                                                                                                  \
    template <class T>                                                                                                 \
    void _save_parent(T& ar) const                                                                                     \
    {                                                                                                                  \
        PARENT1::save(ar);                                                                                             \
    }                                                                                                                  \
    void _bind_parent_slots(bool firstInit) { PARENT1::bindSlots(firstInit); }                                         \
    static void _list_parent_param_info(std::vector<mo::ParamInfo*>& info) { PARENT1::getParamInfoStatic(info); }      \
    static void _list_parent_signals(std::vector<mo::SignalInfo*>& info) { PARENT1::getSignalInfoStatic(info); }       \
    static void _list_parent_slots(std::vector<mo::SlotInfo*>& info) { PARENT1::getSlotInfoStatic(info); }             \
    int _init_parent_signals(bool firstInit) { return PARENT1::initSignals(firstInit); }

#define HANDLE_PARENT_2(PARENT1, PARENT2)                                                                              \
    void _init_parent_params(bool firstInit)                                                                           \
    {                                                                                                                  \
        PARENT1::initParams(firstInit);                                                                                \
        PARENT2::initParams(firstInit);                                                                                \
    }                                                                                                                  \
    void _init_parent_outputs()                                                                                        \
    {                                                                                                                  \
        PARENT1::initOutputs();                                                                                        \
        PARENT2::initOutputs();                                                                                        \
    }                                                                                                                  \
    void _serialize_parent_params(ISimpleSerializer* pSerializer)                                                      \
    {                                                                                                                  \
        PARENT1::serializeParams(pSerializer);                                                                         \
        PARENT2::serializeParams(pSerializer);                                                                         \
    }                                                                                                                  \
    template <class T>                                                                                                 \
    void _load_parent(T& ar)                                                                                           \
    {                                                                                                                  \
        PARENT1::load(ar);                                                                                             \
        PARENT2::load(ar);                                                                                             \
    }                                                                                                                  \
    template <class T>                                                                                                 \
    void _save_parent(T& ar) const                                                                                     \
    {                                                                                                                  \
        PARENT1::save(ar);                                                                                             \
        PARENT2::save(ar);                                                                                             \
    }                                                                                                                  \
    void _bind_parent_slots(bool firstInit)                                                                            \
    {                                                                                                                  \
        PARENT1::bindSlots(firstInit);                                                                                 \
        PARENT2::bindSlots(firstInit);                                                                                 \
    }                                                                                                                  \
    static void _list_parent_param_info(std::vector<mo::ParamInfo*>& info)                                             \
    {                                                                                                                  \
        PARENT1::getParamInfoStatic(info);                                                                             \
        PARENT2::getParamInfoStatic(info);                                                                             \
    }                                                                                                                  \
    static void _list_parent_signals(std::vector<mo::SignalInfo*>& info)                                               \
    {                                                                                                                  \
        PARENT1::getSignalInfoStatic(info);                                                                            \
        PARENT2::getSignalInfoStatic(info);                                                                            \
    }                                                                                                                  \
    static void _list_parent_slots(std::vector<mo::SlotInfo*>& info)                                                   \
    {                                                                                                                  \
        PARENT1::getSlotInfoStatic(info);                                                                              \
        PARENT2::getSlotInfoStatic(info);                                                                              \
    }                                                                                                                  \
    int _init_parent_signals(bool firstInit)                                                                           \
    {                                                                                                                  \
        return PARENT1::initSignals(firstInit) + PARENT2::initSignals(firstInit);                                      \
    }

#ifdef _MSC_VER
#define HANDLE_PARENT(...) BOOST_PP_CAT(BOOST_PP_OVERLOAD(HANDLE_PARENT_, __VA_ARGS__)(__VA_ARGS__), BOOST_PP_EMPTY())
#else
#define HANDLE_PARENT(...) BOOST_PP_OVERLOAD(HANDLE_PARENT_, __VA_ARGS__)(__VA_ARGS__)
#endif

#define HANDLE_NO_PARENT                                                                                               \
    void _init_parent_params(bool firstInit) { (void)firstInit; }                                                      \
    void _init_parent_outputs() {}                                                                                     \
    void _serialize_parent_params(ISimpleSerializer* pSerializer) { (void)pSerializer; }                               \
    template <class T>                                                                                                 \
    void _load_parent(T& ar)                                                                                           \
    {                                                                                                                  \
        (void)ar;                                                                                                      \
    }                                                                                                                  \
    template <class T>                                                                                                 \
    void _save_parent(T& ar) const                                                                                     \
    {                                                                                                                  \
        (void)ar;                                                                                                      \
    }                                                                                                                  \
    void _bind_parent_slots(bool firstInit) { (void)firstInit; }                                                       \
    static void _list_parent_param_info(std::vector<mo::ParamInfo*>& info) { (void)info; }                             \
    static void _list_parent_signals(std::vector<mo::SignalInfo*>& info) { (void)info; }                               \
    static void _list_parent_slots(std::vector<mo::SlotInfo*>& info) { (void)info; }                                   \
    int _init_parent_signals(bool firstInit)                                                                           \
    {                                                                                                                  \
        (void)firstInit;                                                                                               \
        return 0;                                                                                                      \
    }

#define MO_BEGIN_1(CLASS_NAME, N_)                                                                                     \
    typedef CLASS_NAME THIS_CLASS;                                                                                     \
    HANDLE_NO_PARENT                                                                                                   \
    SIGNAL_INFO_START(N_)                                                                                              \
    SIGNALS_START(N_)                                                                                                  \
    SLOT_INFO_START(N_)                                                                                                \
    PARAM_INFO_START(N_)                                                                                               \
    SLOT_START(N_)                                                                                                     \
    PARAM_START(N_)                                                                                                    \
    static rcc::shared_ptr<CLASS_NAME> create();

#define MO_DERIVE_(N_, CLASS_NAME, ...)                                                                                \
    typedef CLASS_NAME THIS_CLASS;                                                                                     \
    HANDLE_PARENT(__VA_ARGS__)                                                                                         \
    SIGNAL_INFO_START(N_)                                                                                              \
    SIGNALS_START(N_)                                                                                                  \
    SLOT_INFO_START(N_)                                                                                                \
    PARAM_INFO_START(N_)                                                                                               \
    SLOT_START(N_)                                                                                                     \
    PARAM_START(N_)                                                                                                    \
    static rcc::shared_ptr<CLASS_NAME> create();

#define MO_END_(N)                                                                                                     \
    SIGNAL_INFO_END(N)                                                                                                 \
    SLOT_INFO_END(N)                                                                                                   \
    PARAM_INFO_END(N)                                                                                                  \
    SIGNALS_END(N)                                                                                                     \
    SLOT_END(N)                                                                                                        \
    PARAM_END(N)

#define MO_ABSTRACT_(N_, CLASS_NAME, ...)                                                                              \
    typedef CLASS_NAME THIS_CLASS;                                                                                     \
    HANDLE_PARENT(__VA_ARGS__);                                                                                        \
    SIGNAL_INFO_START(N_)                                                                                              \
    SIGNALS_START(N_)                                                                                                  \
    SLOT_INFO_START(N_)                                                                                                \
    PARAM_INFO_START(N_)                                                                                               \
    SLOT_START(N_)                                                                                                     \
    PARAM_START(N_)

#define MO_REGISTER_OBJECT(TYPE)                                                                                       \
    static ::mo::MetaObjectInfo<TActual<TYPE>> TYPE##_info;                                                            \
    static ::mo::MetaObjectPolicy<TActual<TYPE>, __COUNTER__, void> TYPE##_policy;                                     \
    ::rcc::shared_ptr<TYPE> TYPE::create()                                                                             \
    {                                                                                                                  \
        auto obj = ::mo::MetaObjectFactory::instance()->create(#TYPE);                                                 \
        return ::rcc::shared_ptr<TYPE>(obj);                                                                           \
    }                                                                                                                  \
    REGISTERCLASS(TYPE, &TYPE##_info);

#define MO_REGISTER_CLASS(TYPE) MO_REGISTER_OBJECT(TYPE)

#else // __CUDACC__
#define MO_REGISTER_OBJECT(TYPE)
#define MO_REGISTER_CLASS(TYPE)
#define MO_BEGIN_1(CLASS, N)
#define MO_BEGIN_2(CLASS, PARENT, N)
#define MO_END_(N)
#endif // __CUDACC__
