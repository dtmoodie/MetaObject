#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/Placeholders.h"
#include "MetaObject/Thread/ThreadRegistry.hpp"
#include "MetaObject/Thread/InterThread.hpp"
#include "MetaObject/Signals/ISignal.hpp"
#include "MetaObject/Signals/Connection.hpp"
#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Context.hpp"
#include <boost/signals2.hpp>

namespace mo
{
    class IMetaObject;
	template<class R, class...T> class TypedSignal<R(T...)> : public ISignal, public boost::signals2::signal<R(T...)>
    {
    public:
        void Disconnect(const std::function<R(T...)>& f)
		{
			boost::signals2::signal<R(T...)>::disconnect(f);
		}
		std::shared_ptr<Connection> Connect(const std::function<R(T...)>& f, size_t destination_thread = GetThisThread(), bool force_queue = false, void* obj = nullptr)
        {
            if(destination_thread != get_this_thread() || force_queue)
            {
                auto f_ = [f, destination_thread, This](T... args)
                {
                    // Lambda function receives the call from the boost signal and then pipes the actual function call
                    // over a queue to the correct thread
                    ThreadSpecificQueue::Push(std::bind([f](T... args_)->void
                    {
                        f(args_...);
                    },args...), destination_thread, obj);
                };
				if(obj == nullptr)
					return std::shared_ptr<Connection>(new Connection(boost::signals2::signal<R(T...)>::connect(f_)));
				else
					return std::shared_ptr<Connection>(new ClassConnection(boost::signals2::signal<R(T...)>::connect(f_), obj));
            }else
            {
                return std::shared_ptr<Connection>(new Connection(boost::signals2::signal<R(T...)>::connect(f)));
            }
        }
        std::shared_ptr<Connection> Connect(const std::function<R(T...)>& f, IMetaObject* obj)
        {
            size_t destination_thread = obj->GetContext()->thread_id;
            if(destination_thread!= get_this_thread() || force_queue)
            {
                auto f_ = [f, destination_thread, This](T... args)
                {
                    // Lambda function receives the call from the boost signal and then pipes the actual function call
                    // over a queue to the correct thread
                    ThreadSpecificQueue::Push(std::bind([f](T... args_)->void
                    {
                        f(args_...);
                    },args...), destination_thread, obj);
                };
				if(This == nullptr)
					return std::shared_ptr<connection>(new connection(boost::signals2::signal<R(T...)>::connect(f_)));
				else
					return std::shared_ptr<connection>(new class_connection(boost::signals2::signal<R(T...)>::connect(f_), obj));
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
