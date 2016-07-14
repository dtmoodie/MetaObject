#pragma once
#include "ISlot.hpp"
#include "TypedCallback.hpp"
#include "TypedSignal.hpp"
#include "MetaObject/Context.hpp"
#include <functional>
#include <future>
namespace mo
{
    template<typename Sig> class TypedSlot{};

    template<typename R, typename... T> class TypedSlot<R(T...)>: public std::function<R(T...)>, public ISlot
    {
    public:
        TypedSlot<R(T...)>& operator=(const std::function<R(T...)>& rhs)
        {
            std::function<R(T...)>::operator=(rhs);
            return *this;
        }


        virtual std::shared_ptr<Connection> Connect(std::weak_ptr<ISignal>& signal)
        {
            auto ptr = signal.lock();
            if(ptr && ptr->GetSignature() == GetSignature())
            {
                auto typed = std::dynamic_pointer_cast<TypedSignal<R(T...)>>(ptr);
                if(typed)
                {
                    // Check context jazz

                    if(_ctx && ptr->_ctx)
                    {
                        if(_ctx->thread_id != ptr->_ctx->thread_id)
                        {
                            auto f_ = [this](T... args)
                            {
                                // Lambda function receives the call from the boost signal and then pipes the actual function call
                                // over a queue to the correct thread
                                
                                ThreadSpecificQueue::Push(std::bind(
                                    [this, &promise](T... args_)->void
                                    {
                                        (*this)(args_...);
                                    },args...), _ctx->thread_id, this);
                            };
					        return std::shared_ptr<Connection>(new Connection(boost::signals2::signal<R(T...)>::connect(f_)));
                        }
                    }
					return std::shared_ptr<Connection>(new Connection(boost::signals2::signal<R(T...)>::connect(f_)));
                }
            }
        }
        

        virtual bool Connect(std::weak_ptr<ICallback>& cb) const
        {
            auto ptr = cb.lock();
            if(ptr && ptr->GetSignature() == GetSignature())
            {
                auto typed = std::dynamic_pointer_cast<TypedCallback<R(T...)>>(ptr);
                if(typed)
                {
                    *typed = *this;
                    return true;
                }
            }
            return false;
        }
        TypeInfo GetSignature() const
        {
            return TypeInfo(typeid(R(T...)));
        }
    };
    template<typename... T> class TypedSlot<void(T...)>: public std::function<void(T...)>, public ISlot
    {
    public:
        TypedSlot<void(T...)>& operator=(const std::function<void(T...)>& rhs)
        {
            std::function<void(T...)>::operator=(rhs);
            return *this;
        }


        virtual std::shared_ptr<Connection> Connect(std::weak_ptr<ISignal>& signal)
        {
            auto ptr = signal.lock();
            if(ptr && ptr->GetSignature() == GetSignature())
            {
                auto typed = std::dynamic_pointer_cast<TypedSignal<void(T...)>>(ptr);
                if(typed)
                {
                    // Check context jazz

                    if(_ctx && ptr->_ctx)
                    {
                        if(_ctx->thread_id != ptr->_ctx->thread_id)
                        {
                            auto f_ = [this](T... args)
                            {
                                // Lambda function receives the call from the boost signal and then pipes the actual function call
                                // over a queue to the correct thread
                                
                                ThreadSpecificQueue::Push(std::bind(
                                    [this](T... args_)->void
                                    {
                                        (*this)(args_...);
                                    },args...), _ctx->thread_id, this);
                            };
					        return std::shared_ptr<Connection>(new Connection(typed->connect(f_)));
                        }
                    }
					return std::shared_ptr<Connection>(new Connection(typed->connect(*this)));
                }
            }
        }
        

        virtual bool Connect(std::weak_ptr<ICallback>& cb) const
        {
            auto ptr = cb.lock();
            if(ptr && ptr->GetSignature() == GetSignature())
            {
                auto typed = std::dynamic_pointer_cast<TypedCallback<void(T...)>>(ptr);
                if(typed)
                {
                    *typed = *this;
                    return true;
                }
            }
            return false;
        }
        TypeInfo GetSignature() const
        {
            return TypeInfo(typeid(void(T...)));
        }
    };

}