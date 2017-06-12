#pragma once

#include <boost/preprocessor/facilities/overload.hpp>
#ifdef _MSC_VER
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/facilities/empty.hpp>
#endif
#include "MetaObject/signals/TSlot.hpp"
#include "MetaObject/core/detail/Counter.hpp"
#include "MetaObject/core/detail/Forward.hpp"
#include "MetaObject/signals/SlotInfo.hpp"

// -------------------------------------------------------------------------------------------
#define SLOT_N(NAME, N, RETURN, ...)\
    virtual RETURN NAME(__VA_ARGS__); \
    mo::TSlot<RETURN(__VA_ARGS__)> COMBINE(_slot_##NAME##_, N); \
    void _bind_slots(bool firstInit, mo::_counter_<N> dummy){ \
        COMBINE(_slot_##NAME##_, N) = my_bind(static_cast<RETURN(THIS_CLASS::*)(__VA_ARGS__)>(&THIS_CLASS::NAME), this, make_int_sequence<BOOST_PP_VARIADIC_SIZE(__VA_ARGS__)>{} ); \
        addSlot(&COMBINE(_slot_##NAME##_, N), #NAME); \
        _bind_slots(firstInit, --dummy); \
    } \
    static void _list_slots(std::vector<mo::SlotInfo*>& info, mo::_counter_<N> dummy){ \
        (void)dummy; \
        _list_slots(info, mo::_counter_<N-1>()); \
        static mo::SlotInfo s_info{mo::TypeInfo(typeid(RETURN(__VA_ARGS__))), #NAME, "", ""}; \
        info.push_back(&s_info); \
    } \
    template<class Sig> mo::TSlot<RETURN(__VA_ARGS__)>* getSlot_##NAME(typename std::enable_if<std::is_same<Sig, RETURN(__VA_ARGS__)>::value>::type* = 0){ \
        return &COMBINE(_slot_##NAME##_,N); \
    }


#define SLOT_1(RETURN, N, NAME) \
    virtual RETURN NAME(); \
    mo::TSlot<RETURN(void)> COMBINE(_slot_##NAME##_, N); \
    void _bind_slots(bool firstInit, mo::_counter_<N> dummy){ \
        COMBINE(_slot_##NAME##_, N) = std::bind(static_cast<RETURN(THIS_CLASS::*)()>(&THIS_CLASS::NAME), this); \
        addSlot(&COMBINE(_slot_##NAME##_, N), #NAME); \
        _bind_slots(firstInit, --dummy); \
    } \
    static void _list_slots(std::vector<mo::SlotInfo*>& info, mo::_counter_<N> dummy){ \
        (void)dummy; \
        _list_slots(info, mo::_counter_<N-1>()); \
        static mo::SlotInfo s_info{mo::TypeInfo(typeid(RETURN(void))), #NAME, "", ""}; \
        info.push_back(&s_info); \
    } \
    template<class Sig> mo::TSlot<RETURN()>* getSlot_##NAME(typename std::enable_if<std::is_same<Sig, RETURN()>::value>::type* = 0){ \
        return &COMBINE(_slot_##NAME##_,N); \
    }

#define SLOT_2(RETURN, N, NAME, ...) SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_3(RETURN, N, NAME, ...) SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_4(RETURN, N, NAME, ...) SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_5(RETURN, N, NAME, ...) SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_6(RETURN, N, NAME, ...) SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_7(RETURN, N, NAME, ...) SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_8(RETURN, N, NAME, ...) SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_9(RETURN, N, NAME, ...) SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_10(RETURN, N, NAME, ...) SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_11(RETURN, N, NAME, ...) SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_12(RETURN, N, NAME, ...) SLOT_N(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_13(RETURN, N, NAME, ...) SLOT_N(NAME, N, RETURN, __VA_ARGS__)



#define DESCRIBE_SLOT_(NAME, DESCRIPTION, N) \
std::string _slot_description_by_name(const std::string& name, mo::_counter_<N> dummy){ \
    (void)dummy; \
    if(name == #NAME) \
        return DESCRIPTION; \
} \
std::vector<slot_info> _list_slots(mo::_counter_<N> dummy){ \
    (void)dummy; \
    auto slot_info = _list_slots(mo::_counter_<N-1>()); \
    for(auto& info : slot_info){ \
        if(info.name == #NAME){ \
            info.description = DESCRIPTION; \
        } \
    } \
    return slot_info; \
}

#define SLOT_TOOLTIP_(name, tooltip, N) \
static void _list_slots(std::vector<mo::SlotInfo*>& info, mo::_counter_<N> dummy){ \
    _list_slots(info, --dummy); \
    for(auto it : info){ \
        if(it->name == #name){ \
            if(it->tooltip.empty()){ \
                it->tooltip = tooltip; \
            } \
        } \
    } \
}

#ifndef __CUDACC__
  #ifdef _MSC_VER
    #define MO_SLOT(RET, ...) BOOST_PP_CAT( BOOST_PP_OVERLOAD(SLOT_, __VA_ARGS__)(RET, __COUNTER__, __VA_ARGS__), BOOST_PP_EMPTY())
  #else
    #define MO_SLOT(NAME, ...) BOOST_PP_OVERLOAD(SLOT_, __VA_ARGS__)(NAME, __COUNTER__, __VA_ARGS__)
  #endif
  #define DESCRIBE_SLOT(NAME, DESCRIPTION) DESCRIBE_SLOT_(NAME, DESCRIPTION, __COUNTER__)
#else
  #define MO_SLOT(RET, ...)
  #define DESCRIBE_SLOT(NAME, DESCRIPTION)
#endif
#define PARAM_UPDATE_SLOT(NAME) MO_SLOT(void, on_##NAME##_modified, mo::IParam*, mo::Context*, mo::OptionalTime_t, size_t, mo::ICoordinateSystem*, mo::UpdateFlags)
#define SLOT_TOOLTIP(name, tooltip) SLOT_TOOLTIP_(name, tooltip, __COUNTER__)
