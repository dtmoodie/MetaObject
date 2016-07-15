#pragma once
#include "MetaObject/Signals/TypedCallback.hpp"
#include "MetaObject/Signals/CallbackInfo.hpp"
#include "MetaObject/Detail/HelperMacros.hpp"



#define MO_CALLBACK_1(RETURN, N, NAME) \
mo::TypedCallback<RETURN(void)> COMBINE(_callback_##NAME##_, N); \
RETURN NAME() \
{ \
    if(COMBINE(_callback_##NAME##_,N)) \
    { \
        return COMBINE(_callback_##NAME##_,N)(); \
    } \
    throw "Callback not set"; \
    return RETURN(); \
} \
void add_callbacks_(mo::_counter_<N> dummy) \
{ \
    AddCallback(&COMBINE(_callback_##NAME##_, N), #NAME); \
    add_callbacks_(--dummy); \
} \
static void list_callbacks_(std::vector<mo::CallbackInfo*>& info, mo::_counter_<N> dummy) \
{ \
    list_callbacks_(info, --dummy); \
    static mo::CallbackInfo s_info{mo::TypeInfo(typeid(RETURN(void))), #NAME}; \
    info.push_back(&s_info); \
}

#define MO_CALLBACK_2(RETURN, N, NAME, ARG1) \
mo::TypedCallback<RETURN(void)> COMBINE(_callback_##NAME##_, N); \
RETURN NAME(ARG1 arg1, ARG2 arg2, ARG3 arg3) \
{ \
    if(COMBINE(_callback_##NAME##_,N)) \
    { \
        return COMBINE(_callback_##NAME##_,N)(arg1); \
    } \
    throw "Callback not set"; \
    return RETURN(); \
} \
void add_callbacks_(mo::_counter_<N> dummy) \
{ \
    AddCallback(&COMBINE(_callback_##NAME##_, N), #NAME); \
} \
static void list_callbacks_(std::vector<mo::CallbackInfo*>& info, mo::_counter_<N> dummy) \
{ \
    list_callbacks_(info, --dummy); \
    static mo::CallbackInfo s_info{mo::TypeInfo(typeid(RETURN(ARG1))), #NAME}; \
    info.push_back(&s_info); \
}

#define MO_CALLBACK_3(RETURN, N, NAME, ARG1, ARG2) \
mo::TypedCallback<RETURN(void)> COMBINE(_callback_##NAME##_, N); \
RETURN NAME(ARG1 arg1, ARG2 arg2, ARG3 arg3) \
{ \
    if(COMBINE(_callback_##NAME##_,N)) \
    { \
        return COMBINE(_callback_##NAME##_,N)(arg1, arg2); \
    } \
    throw "Callback not set"; \
    return RETURN(); \
} \
void add_callbacks_(mo::_counter_<N> dummy) \
{ \
    AddCallback(&COMBINE(_callback_##NAME##_, N), #NAME); \
} \
static void list_callbacks_(std::vector<mo::CallbackInfo*>& info, mo::_counter_<N> dummy) \
{ \
    list_callbacks_(info, --dummy); \
    static mo::CallbackInfo s_info{mo::TypeInfo(typeid(RETURN(ARG1, ARG2))), #NAME}; \
    info.push_back(&s_info); \
}

#define MO_CALLBACK_4(RETURN, N, NAME, ARG1, ARG2, ARG3) \
mo::TypedCallback<RETURN(void)> COMBINE(_callback_##NAME##_, N); \
RETURN NAME(ARG1 arg1, ARG2 arg2, ARG3 arg3, ARG4 arg4) \
{ \
    if(COMBINE(_callback_##NAME##_,N)) \
    { \
        return COMBINE(_callback_##NAME##_,N)(arg1, arg2, arg3); \
    } \
    throw "Callback not set"; \
    return RETURN(); \
} \
void add_callbacks_(mo::_counter_<N> dummy) \
{ \
    AddCallback(&COMBINE(_callback_##NAME##_, N), #NAME); \
} \
static void list_callbacks_(std::vector<mo::CallbackInfo*>& info, mo::_counter_<N> dummy) \
{ \
    list_callbacks_(info, --dummy); \
    static mo::CallbackInfo s_info{mo::TypeInfo(typeid(RETURN(ARG1, ARG2, ARG3))), #NAME}; \
    info.push_back(&s_info); \
}

#define MO_CALLBACK_5(RETURN, N, NAME, ARG1, ARG2, ARG3, ARG4) \
mo::TypedCallback<RETURN(void)> COMBINE(_callback_##NAME##_, N); \
RETURN NAME(ARG1 arg1, ARG2 arg2, ARG3 arg3, ARG4 arg4, ARG5 arg5) \
{ \
    if(COMBINE(_callback_##NAME##_,N)) \
    { \
        return COMBINE(_callback_##NAME##_,N)(arg1, arg2, arg3, arg4); \
    } \
    throw "Callback not set"; \
    return RETURN(); \
} \
void add_callbacks_(mo::_counter_<N> dummy) \
{ \
    AddCallback(&COMBINE(_callback_##NAME##_, N), #NAME); \
} \
static void list_callbacks_(std::vector<mo::CallbackInfo*>& info, mo::_counter_<N> dummy) \
{ \
    list_callbacks_(info, --dummy); \
    static mo::CallbackInfo s_info{mo::TypeInfo(typeid(RETURN(ARG1, ARG2, ARG3, ARG4))), #NAME}; \
    info.push_back(&s_info); \
}

#define MO_CALLBACK_6(RETURN, N, NAME, ARG1, ARG2, ARG3, ARG4, ARG5) \
mo::TypedCallback<RETURN(void)> COMBINE(_callback_##NAME##_, N); \
RETURN NAME(ARG1 arg1, ARG2 arg2, ARG3 arg3, ARG4 arg4, ARG5 arg5) \
{ \
    if(COMBINE(_callback_##NAME##_,N)) \
    { \
        return COMBINE(_callback_##NAME##_,N)(arg1, arg2, arg3, arg4, arg5); \
    } \
    throw "Callback not set"; \
    return RETURN(); \
} \
void add_callbacks_(mo::_counter_<N> dummy) \
{ \
    AddCallback(&COMBINE(_callback_##NAME##_, N), #NAME); \
} \
static void list_callbacks_(std::vector<mo::CallbackInfo*>& info, mo::_counter_<N> dummy) \
{ \
    list_callbacks_(info, --dummy); \
    static mo::CallbackInfo s_info{mo::TypeInfo(typeid(RETURN(ARG1, ARG2, ARG3, ARG4, ARG5))), #NAME}; \
    info.push_back(&s_info); \
}

#define MO_CALLBACK_7(RETURN, N, NAME, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6) \
mo::TypedCallback<RETURN(void)> COMBINE(_callback_##NAME##_, N); \
RETURN NAME(ARG1 arg1, ARG2 arg2, ARG3 arg3, ARG4 arg4, ARG5 arg5, ARG6 arg6) \
{ \
    if(COMBINE(_callback_##NAME##_,N)) \
    { \
        return COMBINE(_callback_##NAME##_,N)(arg1, arg2, arg3, arg4, arg5, arg6); \
    } \
    throw "Callback not set"; \
    return RETURN(); \
} \
void add_callbacks_(mo::_counter_<N> dummy) \
{ \
    AddCallback(&COMBINE(_callback_##NAME##_, N), #NAME); \
} \
static void list_callbacks_(std::vector<mo::CallbackInfo*>& info, mo::_counter_<N> dummy) \
{ \
    list_callbacks_(info, --dummy); \
    static mo::CallbackInfo s_info{mo::TypeInfo(typeid(RETURN(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6))), #NAME}; \
    info.push_back(&s_info); \
}

#define MO_CALLBACK_8(RETURN, N, NAME, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7) \
mo::TypedCallback<RETURN(void)> COMBINE(_callback_##NAME##_, N); \
RETURN NAME(ARG1 arg1, ARG2 arg2, ARG3 arg3, ARG4 arg4, ARG5 arg5, ARG6 arg6, ARG7 arg7) \
{ \
    if(COMBINE(_callback_##NAME##_,N)) \
    { \
        return COMBINE(_callback_##NAME##_,N)(arg1, arg2, arg3, arg4, arg5, arg6, arg7); \
    } \
    throw "Callback not set"; \
    return RETURN(); \
} \
void add_callbacks_(mo::_counter_<N> dummy) \
{ \
    AddCallback(&COMBINE(_callback_##NAME##_, N), #NAME); \
} \
static void list_callbacks_(std::vector<mo::CallbackInfo*>& info, mo::_counter_<N> dummy) \
{ \
    list_callbacks_(info, --dummy); \
    static mo::CallbackInfo s_info{mo::TypeInfo(typeid(RETURN(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7))), #NAME}; \
    info.push_back(&s_info); \
}

#define MO_CALLBACK_9(RETURN, N, NAME, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8) \
mo::TypedCallback<RETURN(void)> COMBINE(_callback_##NAME##_, N); \
RETURN NAME(ARG1 arg1, ARG2 arg2, ARG3 arg3, ARG4 arg4, ARG5 arg5, ARG6 arg6, ARG7 arg7, ARG8 arg8) \
{ \
    if(COMBINE(_callback_##NAME##_,N)) \
    { \
        return COMBINE(_callback_##NAME##_,N)(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); \
    } \
    throw "Callback not set"; \
    return RETURN(); \
} \
void add_callbacks_(mo::_counter_<N> dummy) \
{ \
    AddCallback(&COMBINE(_callback_##NAME##_, N), #NAME); \
} \
static void list_callbacks_(std::vector<mo::CallbackInfo*>& info, mo::_counter_<N> dummy) \
{ \
    list_callbacks_(info, --dummy); \
    static mo::CallbackInfo s_info{mo::TypeInfo(typeid(RETURN(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8))), #NAME}; \
    info.push_back(&s_info); \
}

#define MO_CALLBACK_10(RETURN, N, NAME, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9) \
mo::TypedCallback<RETURN(void)> COMBINE(_callback_##NAME##_, N); \
RETURN NAME(ARG1 arg1, ARG2 arg2, ARG3 arg3, ARG4 arg4, ARG5 arg5, ARG6 arg6, ARG7 arg7, ARG8 arg8, ARG9 arg9) \
{ \
    if(COMBINE(_callback_##NAME##_,N)) \
    { \
        return COMBINE(_callback_##NAME##_,N)(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); \
    } \
    throw "Callback not set"; \
    return RETURN(); \
} \
void add_callbacks_(mo::_counter_<N> dummy) \
{ \
    AddCallback(&COMBINE(_callback_##NAME##_, N), #NAME); \
} \
static void list_callbacks_(std::vector<mo::CallbackInfo*>& info, mo::_counter_<N> dummy) \
{ \
    list_callbacks_(info, --dummy); \
    static mo::CallbackInfo s_info{mo::TypeInfo(typeid(RETURN(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9))), #NAME}; \
    info.push_back(&s_info); \
}

#define MO_CALLBACK_11(RETURN, N, NAME, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9, ARG10) \
mo::TypedCallback<RETURN(void)> COMBINE(_callback_##NAME##_, N); \
RETURN NAME(ARG1 arg1, ARG2 arg2, ARG3 arg3, ARG4 arg4, ARG5 arg5, ARG6 arg6, ARG7 arg7, ARG8 arg8, ARG9 arg9, ARG10 arg10) \
{ \
    if(COMBINE(_callback_##NAME##_,N)) \
    { \
        return COMBINE(_callback_##NAME##_,N)(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10); \
    } \
    throw "Callback not set"; \
    return RETURN(); \
} \
void add_callbacks_(mo::_counter_<N> dummy) \
{ \
    AddCallback(&COMBINE(_callback_##NAME##_, N), #NAME); \
} \
static void list_callbacks_(std::vector<mo::CallbackInfo*>& info, mo::_counter_<N> dummy) \
{ \
    list_callbacks_(info, --dummy); \
    static mo::CallbackInfo s_info{mo::TypeInfo(typeid(RETURN(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9, ARG10))), #NAME}; \
    info.push_back(&s_info); \
}


#ifdef _MSC_VER
#define MO_CALLBACK(RET, ...) BOOST_PP_CAT( BOOST_PP_OVERLOAD(MO_CALLBACK_, __VA_ARGS__)(RET, __COUNTER__, __VA_ARGS__), BOOST_PP_EMPTY())
#else
#define MO_CALLBACK(RET, ...) BOOST_PP_OVERLOAD(MO_CALLBACK_, __VA_ARGS__)(NAME, __COUNTER__, __VA_ARGS__)
#endif