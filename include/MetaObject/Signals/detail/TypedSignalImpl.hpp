#pragma once

namespace mo
{
    template<class Sig> class TypedSignal;
     
    template<class R, class...T> void 
        TypedSignal<R(T...)>::Disconnect(const std::function<R(T...)>& f)
	{
		boost::signals2::signal<R(T...)>::disconnect(f);
	}

	template<class R, class...T> std::shared_ptr<Connection> 
        TypedSignal<R(T...)>::Connect(const std::function<R(T...)>& f, size_t destination_thread = GetThisThread(), bool force_queue = false, void* obj = nullptr)
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

    template<class R, class...T> std::shared_ptr<Connection> 
        TypedSignal<R(T...)>::Connect(const std::function<R(T...)>& f, IMetaObject* obj)
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
				return std::shared_ptr<Connection>(new Connection(boost::signals2::signal<R(T...)>::connect(f_)));
			else
				return std::shared_ptr<Connection>(new ClassConnection(boost::signals2::signal<R(T...)>::connect(f_), obj));
        }else
        {
            return std::shared_ptr<Connection>(new Connection(boost::signals2::signal<R(T...)>::connect(f)));
        }
    }

    template<class R, class...T>std::shared_ptr<Connection> 
        TypedSignal<R(T...)>::Connect(const std::function<R(T...)>& f, int dest_thread_type, bool force_queued, void* This)
    {
        return Connect(f, thread_registry::get_instance()->get_thread(dest_thread_type), force_queued, This);
    }
    template<class R, class...T> std::shared_ptr<Connection> 
        Connect(const std::string& name, SignalManager* mgr)
    {
        
    }

    template<class R, class...T> std::shared_ptr<Connection> 
        Connect(ISlot* slot)
    {
        
    }

    template<class R, class...T> void 
        TypedSignal<R(T...)>::operator()(T... args)
    {
        boost::signals2::signal<R(T...)>::operator()(args...);
    }

    template<class R, class...T> TypeInfo 
        TypedSignal<R(T...)>::GetSignature() const
    {
        return TypeInfo(typeid(R(T...)));
    }
}