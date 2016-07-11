#pragma once
#include "Defs.h"
#include "LokiTypeInfo.h"
#include "thread_registry.h"
#include "connection.h"
#include <memory>

namespace Signals
{
    class signal_sink_base;
    struct signal_info
    {
        Loki::TypeInfo signature;
        std::string name;
        std::string description;
    };
    struct slot_info
    {
        Loki::TypeInfo signature;
        std::string name;
        std::string description;
    };

    class SIGNAL_EXPORTS signal_base
    {
    protected:
		
	public:
        virtual ~signal_base(){}
		virtual void add_log_sink(std::shared_ptr<signal_sink_base> sink, size_t id = get_this_thread()) = 0;
        virtual Loki::TypeInfo get_signal_type() = 0;
    };
}
