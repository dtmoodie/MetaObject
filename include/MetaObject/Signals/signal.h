#pragma once
#include "Defs.h"
#include "placeholder.h"
#include "thread_registry.h"
#include "signal_base.h"
#include "channels.h"
#include "signal_sink_base.h"
#include "signal_sink.h"
#include "connection.h"
#include "combiner.h"
#include "signal_sink_factory.h"
#include "serialization.h"
#include "meta_signal.hpp"
#include <boost/signals2.hpp>
#include <list>
namespace Signals
{
	template<class R, class...T> class typed_signal_base<R(T...)> : public signal_base, public meta_signal<R(T...)>, public boost::signals2::signal<R(T...)>
    {
    protected:
    public:
        virtual void add_log_sink(std::shared_ptr<signal_sink_base> sink, size_t destination_thread = get_this_thread())
        {            
        }
		void disconnect(const std::function<R(T...)>& f)
		{
			boost::signals2::signal<R(T...)>::disconnect(f);
		}
		std::shared_ptr<connection> connect(const std::function<R(T...)>& f, size_t destination_thread = get_this_thread(), bool force_queue = false, void* This = nullptr)
        {
            if(destination_thread != get_this_thread() || force_queue)
            {
                auto f_ = [f, destination_thread, This](T... args)
                {
                    // Lambda function receives the call from the boost signal and then pipes the actual function call
                    // over a queue to the correct thread
                    thread_specific_queue::push(std::bind([f](T... args_)->void
                    {
                        f(args_...);
                    },args...), destination_thread, This);
                };
				if(This == nullptr)
					return std::shared_ptr<connection>(new connection(boost::signals2::signal<R(T...)>::connect(f_)));
				else
					return std::shared_ptr<connection>(new class_connection(boost::signals2::signal<R(T...)>::connect(f_), This));
            }else
            {
                return std::shared_ptr<connection>(new connection(boost::signals2::signal<R(T...)>::connect(f)));
            }
        }
		std::shared_ptr<connection> connect_log_sink(const std::function<void(T...)>& f, size_t destination_thread = get_this_thread())
		{
            return connect(f, destination_thread);
		}

        std::shared_ptr<connection> connect(const std::function<R(T...)>& f, int dest_thread_type, bool force_queued = false, void* This = nullptr)
        {
            return connect(f, thread_registry::get_instance()->get_thread(dest_thread_type), force_queued, This);
        }
        void operator()(T... args)
        {
            boost::signals2::signal<R(T...)>::operator()(args...);
        }
        virtual Loki::TypeInfo get_signal_type()
        {
            return Loki::TypeInfo(typeid(R(T...)));
        }
    };
}
