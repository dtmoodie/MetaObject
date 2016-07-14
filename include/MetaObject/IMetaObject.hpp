#pragma once
#include <IObject.h>
#include "MetaObject/Detail/TypeInfo.h"
#include <memory>
namespace mo
{
    class Context;
    class SignalManager;
    class ISignal;
    class ICallback;
    class Connection;
    
    class IParameter;
    template<class T> class ITypedParameter;

    struct ParameterInfo;
    struct SignalInfo;
    struct SlotInfo;
    struct CallbackInfo;
    /*
      The IMetaObject interface class defines interfaces for introspection and serialization
      A IMetaObject derivative should use the IMetaObject macros for defining parameters, signals,
      slots, and callbacks.
      parameters - Outputs, Inputs, Control, and State.
       - Outputs parameters are shared with other IMetaObjects
       - Inputs parameters are read from other IMetaObjects
       - Control parameters are user set settings
       - State parameters are used status introspection
     Signals
      - functions that are called by an IMetaObject that invoke all connected slots
      - must have void return type
      - must handle asynchronous operation
     Slots
      - functions that are called when a signal is invoked
      - must have void return type
      - should be called on the thread of the owning context
     Callbacks
      - similar to signals but must have 1 to 1 mapping
      - calls slots with non void return type, returns a std::future of the expected return type
      - called slot could be called asynchronously if it is from an object that is in a different context

      A little detail of how callbacks work.
      When a callback is connected from a sender to a receiver, a ICallback* is passed to the receiver.
      The signature of this callback is checked via the GetSignature() function in the ConnectCallback_
      implementation functions for each slot.  If a signature match is found and a name match is fond
      then the ICallback* is casted to a TypedCallback<Signature> and then it is bound to the slot function

      How signals work.
      Signals are owned by a SignalManager object, the signal manager owns all signals, which are of type
      TypedSignal which wraps boost::signals::signal2.  Any object that calls a signal will hold a pointer
      to the TypedSignal object that it will call.  Any call of a signal will then be routed to all slots
      connected to that signal.

    */


    class IMetaObject: public IObject
    {
    public:
        IMetaObject();
        virtual ~IMetaObject();
        
        // Setup
        virtual void SetContext(Context* ctx);
        Context* GetContext() const;
        virtual void SetupSignals(SignalManager* mgr) = 0;

        // ------- Introspection
        // Get vector of info objects for each corresponding introspection class
        // optional name parameter can be used to get information for corresponding exact matches
        virtual std::vector<ParameterInfo*> GetParameterInfo(const std::string& name = "") const = 0;
        virtual std::vector<SignalInfo*>    GetSignalInfo(const std::string& name = "") const = 0;
        virtual std::vector<SlotInfo*>      GetSlotInfo(const std::string& name = "") const = 0;
        virtual std::vector<CallbackInfo*>  GetCallbackInfo(const std::string& name = "") const = 0;
        
        // -------- Signals / slots
        virtual bool ConnectByName(const std::string& name, ISignal* sig) = 0;
        virtual int  ConnectByName(const std::string& name, SignalManager* mgr) = 0;
        virtual int  ConnectAll(SignalManager* mgr) = 0;
        virtual std::vector<ISignal*> GetAllSignals() const = 0;
        virtual std::vector<ISignal*> GetSignals(const std::string& name) const = 0;
        virtual std::vector<ISignal*> GetSignals(TypeInfo type) const = 0;
        
        virtual int  ConnectByName(const std::string& name, IMetaObject* obj) = 0;

        // Connects all callbacks with callback_name to any accepting slot_name in the other IMetaObject
        // This allows mismatch of callback / slot names.
        virtual int ConnectCallbacks(const std::string& callback_name, const std::string& slot_name, IMetaObject* obj) = 0;

        // Given input callback, connect to any matching slot with name
        // Return true on success, false on failure.  False should only occur if no callback exist with
        // the given name with the correct signature
        // force_queue will create a queued connection such that when the callback is called, the corresponding
        // slot on this object will be called when the event loop of _ctx->thread_id is called.
        virtual bool ConnectCallback(ICallback* callback, const std::string& name, bool force_queue = false) = 0;

        // Connects all callbacks by name of this object to all slots of corresponding name and
        // type in the other object
        // Requires exact name and signature match to connect
        virtual int ConnectCallbacks(IMetaObject* obj) = 0;

        virtual int  DisconnectByName(const std::string& name) = 0;
        virtual bool Disconnect(ISignal* sig);
        virtual int  Disconnect(IMetaObject* obj);

        // Parameters
        virtual std::vector<std::weak_ptr<IParameter>> GetDisplayParameters() const;
        std::weak_ptr<IParameter> GetParameter(const std::string& name) const;
        std::weak_ptr<IParameter> GetParameterOptional(const std::string& name) const;
        std::weak_ptr<IParameter> AddParameter(std::shared_ptr<IParameter> param);
        template<class T> std::weak_ptr<ITypedParameter<T>> GetParameter(const std::string& name) const;

        
    protected:
        virtual void AddConnection(std::shared_ptr<Connection>& connection, ISignal* sig);
        Context* _ctx;
        std::vector<std::shared_ptr<IParameter>> _explicit_parameters;
        std::vector<ICallback*> _explicit_callbacks;
        SignalManager* _sig_manager;
    private:
        struct impl;
        impl* _pimpl;
    };
}