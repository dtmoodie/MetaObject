// -------------------------------------------------------------------------------------------
#define SLOT__(NAME, N, RETURN, ...)\
    virtual RETURN NAME(__VA_ARGS__); \
    template<typename DUMMY> struct slot_registerer<N, DUMMY> \
    { \
        static void RegisterStatic(Signals::signal_registry* registry) \
        { \
            registry->add_slot(Loki::TypeInfo(typeid(THIS_CLASS)),#NAME, Loki::TypeInfo(typeid(RETURN(__VA_ARGS__)))); \
            slot_registerer<N-1, DUMMY>::RegisterStatic(registry); \
        } \
    }; \
    bool connect_(std::string name, Signals::signal_base* signal, Signals::_counter_<N> dummy) \
    {   \
        if(name == #NAME) \
        { \
            auto typed_signal = dynamic_cast<Signals::typed_signal_base<RETURN(__VA_ARGS__)>*>(signal); \
            if(typed_signal) \
            { \
				LOG(trace) << "[" #NAME " - " << typed_signal->get_signal_type().name() << "]"; \
                _connections[signal] = typed_signal->connect(my_bind((RETURN(THIS_CLASS::*)(__VA_ARGS__))&THIS_CLASS::NAME, this, make_int_sequence<BOOST_PP_VARIADIC_SIZE(__VA_ARGS__)>{} )); \
                return true; \
            } \
        } \
        return connect_(name, signal, dummy--); \
    } \
	int connect_(std::string name, manager* manager_, Signals::_counter_<N> dummy) \
	{ \
		if(name == #NAME) \
		{ \
			auto sig = manager_->get_signal<RETURN(__VA_ARGS__)>(#NAME); \
			LOG(trace) << "Connecting slot with name: \"" #NAME "\" and signature <" << typeid(RETURN(__VA_ARGS__)).name() << "> to manager"; \
			_connections[sig] = sig->connect(my_bind((RETURN(THIS_CLASS::*)(__VA_ARGS__))&THIS_CLASS::NAME, this, make_int_sequence<BOOST_PP_VARIADIC_SIZE(__VA_ARGS__)>{} )); \
            manager_->register_receiver(Loki::TypeInfo(typeid(RETURN(__VA_ARGS__))), #NAME, this); \
			return connect_(name, manager_, Signals::_counter_<N-1>()) + 1; \
		} \
		return connect_(name, manager_, Signals::_counter_<N-1>()); \
	} \
	int disconnect_by_name(std::string name, manager* manager_, Signals::_counter_<N> dummy) \
	{ \
		if(name == #NAME) \
		{ \
			auto sig = manager_->get_signal_optional<RETURN(__VA_ARGS__)>(#NAME); \
			if(sig)  \
            { \
                LOG(trace) << "[" #NAME " - " << sig->get_signal_type().name() << "]"; \
                manager_->remove_sender(this, #NAME); \
            } \
			return disconnect_by_name(name, manager_, Signals::_counter_<N-1>()) + disconnect_from_signal(sig) ? 1 : 0; \
		} \
		return disconnect_by_name(name, manager_, Signals::_counter_<N-1>()); \
	} \
	int disconnect_(manager* manager_, Signals::_counter_<N> dummy) \
	{ \
		auto sig = manager_->get_signal_optional<RETURN(__VA_ARGS__)>(#NAME); \
		if(sig) \
        { \
            LOG(trace) << "[" #NAME " - " << sig->get_signal_type().name() << "]"; \
            manager_->remove_sender(this, #NAME); \
        } \
		return disconnect_(manager_, Signals::_counter_<N-1>()) + (disconnect_from_signal(sig) ? 1 : 0); \
	} \
    static std::vector<Signals::slot_info> list_slots_(Signals::_counter_<N> dummy) \
    { \
        auto slot_infos = list_slots_(Signals::_counter_<N-1>()); \
        slot_infos.push_back(Signals::slot_info({Loki::TypeInfo(typeid(RETURN(__VA_ARGS__))), #NAME})); \
        return slot_infos; \
    }

#define SLOT_1(RETURN, N, NAME) \
    virtual RETURN NAME(); \
    template<typename DUMMY> struct slot_registerer<N, DUMMY> \
    { \
        static void RegisterStatic(Signals::signal_registry* registry) \
        { \
            registry->add_slot(Loki::TypeInfo(typeid(THIS_CLASS)),#NAME, Loki::TypeInfo(typeid(RETURN()))); \
            slot_registerer<N-1, DUMMY>::RegisterStatic(registry); \
        } \
    }; \
    bool connect_(std::string name, Signals::signal_base* signal, Signals::_counter_<N> dummy) \
    {   \
        if(name == #NAME) \
        { \
            auto typed_signal = dynamic_cast<Signals::typed_signal_base<RETURN()>*>(signal); \
            if(typed_signal) \
            { \
				LOG(trace) << "[" #NAME  " - " << typed_signal->get_signal_type().name() << "]"; \
                _connections[signal] = typed_signal->connect(std::bind((RETURN(THIS_CLASS::*)())&THIS_CLASS::NAME, this)); \
                return true; \
            } \
        } \
        return connect_(name, signal, Signals::_counter_<N-1>()); \
    } \
    int connect_(std::string name, manager* manager_, Signals::_counter_<N> dummy) \
    { \
        if(name == #NAME)  \
        { \
            auto sig = manager_->get_signal<RETURN()>(#NAME); \
			LOG(trace) << "Connecting slot with name: \"" #NAME "\" and signature <" << typeid(RETURN(void)).name() << "> to manager"; \
            _connections[sig] = sig->connect(std::bind((RETURN(THIS_CLASS::*)())&THIS_CLASS::NAME, this)); \
            manager_->register_receiver(Loki::TypeInfo(typeid(RETURN(void))), #NAME, this); \
			return connect_(name, manager_, Signals::_counter_<N-1>()) + 1; \
        } \
        return connect_(name, manager_, Signals::_counter_<N-1>()); \
    } \
	int disconnect_by_name(std::string name, manager* manager_, Signals::_counter_<N> dummy) \
	{ \
		if(name == #NAME) \
		{ \
			auto sig = manager_->get_signal_optional<RETURN()>(#NAME); \
            if(sig) \
            { \
                LOG(trace) << "[" #NAME  " - " << sig->get_signal_type().name() << "]"; \
                manager_->remove_sender(this, #NAME); \
            } \
            return disconnect_by_name(name, manager_, Signals::_counter_<N-1>()) + disconnect_from_signal(sig) ? 1 : 0; \
		} \
		return disconnect_by_name(name, manager_, Signals::_counter_<N-1>()); \
	} \
	int disconnect_(manager* manager_, Signals::_counter_<N> dummy) \
	{ \
		auto sig = manager_->get_signal_optional<RETURN()>(#NAME); \
		if(sig) \
        { \
            LOG(trace) << "[" #NAME  " - " << sig->get_signal_type().name() << "]"; \
            manager_->remove_sender(this, #NAME); \
        } \
        return disconnect_(manager_, Signals::_counter_<N-1>()) + (disconnect_from_signal(sig) ? 1 : 0); \
	} \
    static std::vector<Signals::slot_info> list_slots_(Signals::_counter_<N> dummy) \
    { \
        auto slot_infos = list_slots_(Signals::_counter_<N-1>()); \
        slot_infos.push_back(Signals::slot_info({Loki::TypeInfo(typeid(RETURN(void))), #NAME})); \
        return slot_infos; \
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
std::string slot_description_by_name_(const std::string& name, Signals::_counter_<N> dummy) \
{ \
    if(name == #NAME) \
        return DESCRIPTION; \
} \
std::vector<slot_info> list_slots_(Signals::_counter_<N> dummy) \
{ \
    auto slot_info = list_slots_(Signals::_counter_<N-1>()); \
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
int connect_(Signals::signal_manager* manager, Signals::_counter_<N> dummy) \
{ \
	LOG(trace) << "Automatically connecting slot named: \"" #NAME "\" to manager"; \
    int count = connect_(#NAME, manager, Signals::_counter_<N>()); \
    return connect_(manager, Signals::_counter_<N-1>()) + count; \
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
