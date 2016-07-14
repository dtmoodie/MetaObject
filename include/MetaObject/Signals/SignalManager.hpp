#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include <mutex>
#include <memory>
#include <vector>

namespace mo
{    
    struct receiver
    {
        void* ptr; // Pointer to object   
        TypeInfo type; // Object type
        TypeInfo signature; // Signal signature
        std::string signal_name;// Signal name
        std::string description; // sender description
        std::string file; // File from which the sender is sending the signal, mostly used when not associated to an object
        int line; // Line from which the signal is sent
        std::string tooltip; // description of what happens when this signal is received
    };

    struct sender
    {
        void* ptr; // Pointer to object
        TypeInfo type; // Object type
        TypeInfo signature; // Signal signature
        std::string signal_name; // Signal name
        std::string description; // sender description
        std::string tooltip; // Description of why this signal is sent
        std::string file; // File from which the sender is sending the signal, mostly used when not associated to an object
        int line; // Line from which the signal is sent
    };
    
	class signaler;
    class ISignal;
    class IMetaObject;
    template<class Sig> class TypedSignal;

    // Manages shared ownership of signals so that multiple senders and receivers can exist, also allows for encapsulation of groups of signals based on subsystem
    class MO_EXPORTS SignalManager
    {
    public:
        SignalManager();
        ~SignalManager();
        static SignalManager* Instance();
        static void SetInstance(SignalManager* inst);
		
        template<typename T> std::weak_ptr<TypedSignal<T>> GetSignalOptional(const std::string& name);

		template<typename T> std::weak_ptr<TypedSignal<T>> GetSignal(const std::string& name);

		virtual std::vector<std::weak_ptr<ISignal>> GetSignals(std::string name);
		virtual std::vector<std::weak_ptr<ISignal>> GetSignals(TypeInfo type);
		virtual std::vector<std::weak_ptr<ISignal>> GetSignals(TypeInfo type, std::string name);
		virtual std::vector<std::weak_ptr<ISignal>> GetSignals();
        virtual std::vector<std::string> GetSignalNames();
		virtual void PrintSignalMap();
        
        // Helps with dynamic introspection on what actually is connected
        /*virtual std::vector<receiver> GetReceivers(TypeInfo type, std::string name);
        virtual std::vector<receiver> GetReceivers(TypeInfo type);
        virtual std::vector<receiver> GetReceivers(std::string name);
        virtual std::vector<receiver> GetReceivers();

        virtual std::vector<sender> GetSenders(TypeInfo type, std::string name);
        virtual std::vector<sender> GetSenders(TypeInfo type);
        virtual std::vector<sender> GetSenders(std::string name);
        virtual std::vector<sender> GetSenders();*/

        virtual std::weak_ptr<ISignal> GetSignalOptional(const std::string& name, const TypeInfo& type);
        virtual std::weak_ptr<ISignal> GetSignalOptional(const std::string& name, const std::string& type);

    protected:
        virtual std::shared_ptr<ISignal>& GetSignal(const std::string& name, TypeInfo type);
        bool exists(const std::string& name, TypeInfo type);
    private:
        struct impl;
        impl* _pimpl;
        std::mutex mtx;
    };
} // namespace Signals
