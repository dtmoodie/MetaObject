#include "MetaObject/Signals/TypedSlot.hpp"
#include "MetaObject/Detail/HelperMacros.hpp"
#include "MetaObject/Signals/SlotInfo.hpp"
// -------------------------------------------------------------------------------------------
#define SLOT__(NAME, N, RETURN, ...)\
    virtual RETURN NAME(__VA_ARGS__); \
    bool connect_(std::string name, mo::ISignal* signal, mo::_counter_<N> dummy) \
    {   \
        if(name == #NAME) \
        { \
            auto TypedSignal = dynamic_cast<mo::TypedSignal<RETURN(__VA_ARGS__)>*>(signal); \
            if(TypedSignal) \
            { \
				LOG(trace) << "[" #NAME " - " << TypedSignal->GetSignature().name() << "]"; \
                _connections[signal] = TypedSignal->connect(my_bind((RETURN(THIS_CLASS::*)(__VA_ARGS__))&THIS_CLASS::NAME, this, make_int_sequence<BOOST_PP_VARIADIC_SIZE(__VA_ARGS__)>{} )); \
                return true; \
            } \
        } \
        return connect_(name, signal, dummy--); \
    } \
	int connect_(std::string name, manager* manager_, mo::_counter_<N> dummy) \
	{ \
		if(name == #NAME) \
		{ \
			auto sig = manager_->GetSignal<RETURN(__VA_ARGS__)>(#NAME); \
			LOG(trace) << "Connecting slot with name: \"" #NAME "\" and signature <" << typeid(RETURN(__VA_ARGS__)).name() << "> to manager"; \
			_connections[sig] = sig->connect(my_bind((RETURN(THIS_CLASS::*)(__VA_ARGS__))&THIS_CLASS::NAME, this, make_int_sequence<BOOST_PP_VARIADIC_SIZE(__VA_ARGS__)>{} )); \
            manager_->register_receiver(Loki::TypeInfo(typeid(RETURN(__VA_ARGS__))), #NAME, this); \
			return connect_(name, manager_, mo::_counter_<N-1>()) + 1; \
		} \
		return connect_(name, manager_, mo::_counter_<N-1>()); \
	} \
	int disconnect_by_name(std::string name, manager* manager_, mo::_counter_<N> dummy) \
	{ \
		if(name == #NAME) \
		{ \
			auto sig = manager_->get_signal_optional<RETURN(__VA_ARGS__)>(#NAME); \
			if(sig)  \
            { \
                LOG(trace) << "[" #NAME " - " << sig->GetSignature().name() << "]"; \
                manager_->remove_sender(this, #NAME); \
            } \
			return disconnect_by_name(name, manager_, mo::_counter_<N-1>()) + disconnect_from_signal(sig) ? 1 : 0; \
		} \
		return disconnect_by_name(name, manager_, mo::_counter_<N-1>()); \
	} \
	int disconnect_(manager* manager_, mo::_counter_<N> dummy) \
	{ \
		auto sig = manager_->get_signal_optional<RETURN(__VA_ARGS__)>(#NAME); \
		if(sig) \
        { \
            LOG(trace) << "[" #NAME " - " << sig->GetSignature().name() << "]"; \
            manager_->remove_sender(this, #NAME); \
        } \
		return disconnect_(manager_, mo::_counter_<N-1>()) + (disconnect_from_signal(sig) ? 1 : 0); \
	} \
    static std::vector<mo::slot_info> list_slots_(mo::_counter_<N> dummy) \
    { \
        auto slot_infos = list_slots_(mo::_counter_<N-1>()); \
        slot_infos.push_back(mo::slot_info({Loki::TypeInfo(typeid(RETURN(__VA_ARGS__))), #NAME})); \
        return slot_infos; \
    }

#define SLOT_1(RETURN, N, NAME) \
    virtual RETURN NAME(); \
    mo::TypedSlot<RETURN(void)> COMBINE(_slot_##NAME##_, N); \
    void bind_slots_(mo::_counter_<N> dummy) \
    { \
        COMBINE(_slot_##NAME##_, N) = std::bind((RETURN(THIS_CLASS::*)())&THIS_CLASS::NAME, this); \
        AddSlot(&COMBINE(_slot_##NAME##_, N), #NAME); \
    } \
    static void list_slots_(std::vector<mo::SlotInfo*>& info, mo::_counter_<N> dummy) \
    { \
        list_slots_(info, mo::_counter_<N-1>()); \
        static mo::SlotInfo s_info{mo::TypeInfo(typeid(RETURN(void))), #NAME}; \
        info.push_back(&s_info); \
    }

#define SLOT_2(RETURN, N, NAME, ...) SLOT__(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_3(RETURN, N, NAME, ...) SLOT__(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_4(RETURN, N, NAME, ...) SLOT__(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_5(RETURN, N, NAME, ...) SLOT__(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_6(RETURN, N, NAME, ...) SLOT__(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_7(RETURN, N, NAME, ...) SLOT__(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_8(RETURN, N, NAME, ...) SLOT__(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_9(RETURN, N, NAME, ...) SLOT__(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_10(RETURN, N, NAME, ...) SLOT__(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_11(RETURN, N, NAME, ...) SLOT__(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_12(RETURN, N, NAME, ...) SLOT__(NAME, N, RETURN, __VA_ARGS__)
#define SLOT_13(RETURN, N, NAME, ...) SLOT__(NAME, N, RETURN, __VA_ARGS__)



#define DESCRIBE_SLOT_(NAME, DESCRIPTION, N) \
std::string slot_description_by_name_(const std::string& name, mo::_counter_<N> dummy) \
{ \
    if(name == #NAME) \
        return DESCRIPTION; \
} \
std::vector<slot_info> list_slots_(mo::_counter_<N> dummy) \
{ \
    auto slot_info = list_slots_(mo::_counter_<N-1>()); \
    for(auto& info : slot_info) \
    { \
        if(info.name == #NAME) \
        { \
            info.description = DESCRIPTION; \
        } \
    } \
    return slot_info; \
}\

#define DESCRIBE_SLOT(NAME, DESCRIPTION) DESCRIBE_SLOT_(NAME, DESCRIPTION, __COUNTER__)

#define REGISTER_SLOT_(NAME, N) \
int connect_(mo::signal_manager* manager, mo::_counter_<N> dummy) \
{ \
	LOG(trace) << "Automatically connecting slot named: \"" #NAME "\" to manager"; \
    int count = connect_(#NAME, manager, mo::_counter_<N>()); \
    return connect_(manager, mo::_counter_<N-1>()) + count; \
}
// -------------------------------------------------------------------------------------------
// This macro signifies that this slot should automatically be connected to the appropriate signal from the signal_manager that is passed into setup_signals


#define REGISTER_SLOT(NAME) REGISTER_SLOT_(NAME, __COUNTER__)

#ifdef _MSC_VER
#define SLOT_DEF(RET, ...) BOOST_PP_CAT( BOOST_PP_OVERLOAD(SLOT_, __VA_ARGS__)(RET, __COUNTER__, __VA_ARGS__), BOOST_PP_EMPTY())
#else
#define SLOT_DEF(NAME, ...) BOOST_PP_OVERLOAD(SLOT_, __VA_ARGS__)(NAME, __COUNTER__, __VA_ARGS__)
#endif

#ifdef _MSC_VER
#define SLOT_OVERLOAD(NAME, ...) BOOST_PP_CAT(BOOST_PP_OVERLOAD(SLOT_OVERLOAD_, __VA_ARGS__)(NAME, __COUNTER__, __VA_ARGS__), BOOST_PP_EMPTY())
#else
#define SLOT_OVERLOAD(NAME, ...) BOOST_PP_OVERLOAD(SLOT_OVERLOAD_, __VA_ARGS__)(NAME, __COUNTER__, __VA_ARGS__)
#endif

#define CALL_REGISTER_SLOT(name) REGISTER_SLOT(name)
#define REGISTER_SLOT_HELPER(NAME, ...) CALL_REGISTER_SLOT(NAME);

#define AUTO_SLOT(RETURN, ...) \
SLOT_DEF(RETURN, __VA_ARGS__); \
REGISTER_SLOT_HELPER(__VA_ARGS__)
