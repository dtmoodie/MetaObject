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
    template<class Sig> class TypedSignal{};

	template<class R, class...T> class TypedSignal<R(T...)> : public ISignal, public boost::signals2::signal<R(T...)>
    {
    public:
		std::shared_ptr<Connection> Connect(const std::function<R(T...)>& f, size_t destination_thread = GetThisThread(), bool force_queue = false, void* obj = nullptr);
        std::shared_ptr<Connection> Connect(const std::function<R(T...)>& f, IMetaObject* obj);
        std::shared_ptr<Connection> Connect(const std::function<R(T...)>& f, int dest_thread_type, bool force_queued = false, void* obj = nullptr);
        std::shared_ptr<Connection> Connect(const std::string& name, SignalManager* mgr);
        std::shared_ptr<Connection> Connect(ISlot* slot) = 0;

        void Disconnect(const std::function<R(T...)>& f);
        void Disconnect(const std::string& name, SignalManager* mgr);
        void Disconnect(ISlot* slot);

        void operator()(T... args)
        TypeInfo GetSignature() const
    };
}
