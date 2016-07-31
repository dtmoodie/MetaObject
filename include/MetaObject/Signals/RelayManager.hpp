#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include <mutex>
#include <memory>
#include <vector>

namespace mo
{
    class ISignal;
	class ISlot;
    class IMetaObject;
	class ISignalRelay;
	class Connection;
	template<typename T> class TypedSignal;
	template<typename T> class TypedSlot;
	template<typename T> class TypedRelay;
    
    // Manages shared ownership of signals so that multiple senders and receivers can exist, also allows for encapsulation of groups of signals based on subsystem
    class MO_EXPORTS RelayManager
    {
    public:
		RelayManager();
        ~RelayManager();
        static RelayManager* Instance();
        static void SetInstance(RelayManager* inst);
		
		std::shared_ptr<Connection> Connect(ISlot* slot, const std::string& name, IMetaObject* obj = nullptr);
		std::shared_ptr<Connection> Connect(ISignal* signal, const std::string& name, IMetaObject* obj = nullptr);

		bool ConnectSignal(IMetaObject* obj, const std::string& name, const TypeInfo& type);
		int  ConnectSignals(IMetaObject* obj, const std::string& name);
		int  ConnectSignals(IMetaObject* obj, const TypeInfo& type);
		int  ConnectSignals(IMetaObject* obj);

		bool ConnectSlot(IMetaObject* obj, const std::string& name, const TypeInfo& type);
		int  ConnectSlots(IMetaObject* obj, const std::string& name);
		int  ConnectSlots(IMetaObject* obj, const TypeInfo& type);
		int  ConnectSlots(IMetaObject* obj);
    protected:
		std::shared_ptr<ISignalRelay>& GetRelay(const TypeInfo& type, const std::string& name);
        bool exists(const std::string& name, TypeInfo type);
    private:
        struct impl;
        impl* _pimpl;
        std::mutex mtx;
    };
} // namespace Signals
