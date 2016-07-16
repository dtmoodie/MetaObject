#pragma once
#include <IObject.h>
#include "MetaObject/Detail/TypeInfo.h"
#include "MetaObject/Detail/Export.hpp"
#include <memory>
namespace boost
{
    namespace signals2
    {
        class connection;
    }
}

namespace mo
{
    class Context;
    class SignalManager;
    class ISignal;
    class ICallback;
    class ISlot;
    class Connection;
    
    class IParameter;
    class InputParameter;
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


    class MO_EXPORTS IMetaObject: public IObject
    {
    public:
        IMetaObject();
        virtual ~IMetaObject();
        
        // Setup
        virtual void SetContext(Context* ctx);
        Context* GetContext() const;
        virtual int SetupSignals(SignalManager* mgr) = 0;
        virtual void BindSlots() = 0;
        virtual void Init(bool firstInit);
        virtual void InitParameters(bool firstInit) = 0;
        virtual void InitCallbacks(bool firstInit) = 0;

        // ------- Introspection
        // Get vector of info objects for each corresponding introspection class
        // optional name parameter can be used to get information for corresponding exact matches
        virtual std::vector<ParameterInfo*> GetParameterInfo() const = 0;
        virtual std::vector<ParameterInfo*> GetParameterInfo(const std::string& name) const;

        virtual std::vector<SignalInfo*>    GetSignalInfo() const = 0;
        virtual std::vector<SignalInfo*>    GetSignalInfo(const std::string& name) const;
        
        virtual std::vector<SlotInfo*>      GetSlotInfo() const = 0;
        virtual std::vector<SlotInfo*>      GetSlotInfo(const std::string& name) const;

        virtual std::vector<CallbackInfo*>  GetCallbackInfo() const = 0;
        virtual std::vector<CallbackInfo*>  GetCallbackInfo(const std::string& name) const;
        
        // -------- Signals / slots
        // If this class emits a signal by the given name, then the input sig will be added to the list of signals
        // that will be called when the signal is emitted.
        virtual bool ConnectByName(const std::string& name, std::weak_ptr<ISignal> sig) = 0;

        // Be careful to only call this once for each mgr object
        // This will call GetSignal<>(name) on the input mgr object and add the obtained signal
        // To the list of signals that is called whenever sig_{name} is emitted
        virtual int  ConnectByName(const std::string& name, SignalManager* mgr) = 0;

        // Be careful to only call once for each mgr object
        // This will call mgr->GetSignal<>() for each declared signal
        virtual int  ConnectAll(SignalManager* mgr);


        virtual std::vector<std::shared_ptr<ISignal>> GetSignals() const;
        virtual std::vector<std::shared_ptr<ISignal>> GetSignals(const std::string& name) const;
        virtual std::vector<std::shared_ptr<ISignal>> GetSignals(const TypeInfo& type) const;

        virtual std::vector<ISlot*> GetSlots() const;
        virtual std::vector<ISlot*> GetSlots(const std::string& name) const;
        virtual std::vector<ISlot*> GetSlots(const TypeInfo& signature) const;
        virtual ISlot* GetSlot(const std::string& name, const TypeInfo& signature) const;

        virtual std::vector<ICallback*> GetCallbacks() const;
        virtual std::vector<ICallback*> GetCallbacks(const std::string& name) const;
        virtual std::vector<ICallback*> GetCallbacks(const TypeInfo& signature) const;
        virtual ICallback* GetCallback(const std::string& name, const TypeInfo& signature) const;


        
        virtual int  ConnectByName(const std::string& name, IMetaObject* obj);

        // Connects all callbacks with callback_name to any accepting slot_name in the other IMetaObject
        // This allows mismatch of callback / slot names.
        int ConnectCallbacks(const std::string& callback_name, const std::string& slot_name, IMetaObject* slot_owner, bool force_queue = false);
        bool ConnectCallback(const TypeInfo& signature, const std::string& callback_name, const std::string& slot_name, IMetaObject* slot_owner, bool force_queue = false);

        // Given input callback, connect to any matching slot with name
        // Return true on success, false on failure.  False should only occur if no callback exist with
        // the given name with the correct signature
        // force_queue will create a queued connection such that when the callback is called, the corresponding
        // slot on this object will be called when the event loop of _ctx->thread_id is called.
        bool ConnectCallback(ICallback* callback, const std::string& name, bool force_queue = false);
        bool ConnectCallback(ISlot* slot, const std::string& callback_name, bool force_queue = false);

        // Connects all callbacks by name of this object to all slots of corresponding name and
        // type in the other object
        // Requires exact name and signature match to connect
        int ConnectCallbacks(IMetaObject* obj, bool force_queue = false);


        virtual int  DisconnectByName(const std::string& name);
        virtual bool Disconnect(ISignal* sig);
        virtual int  Disconnect(IMetaObject* obj);

        // Parameters
        virtual std::vector<IParameter*> GetDisplayParameters() const;
        
        /*std::weak_ptr<IParameter> GetParameter(const std::string& name) const;
        std::weak_ptr<IParameter> GetParameterOptional(const std::string& name) const;*/
        std::vector<InputParameter*> GetInputs(const std::string& name_filter = "") const;
        std::vector<InputParameter*> GetInputs(const TypeInfo& type_filter) const;

        IParameter* GetParameter(const std::string& name) const;
        IParameter* GetParameterOptional(const std::string& name) const;
        std::vector<IParameter*> GetParameters(const std::string& filter) const;
        std::vector<IParameter*> GetParameters(const TypeInfo& filter) const;

        std::weak_ptr<IParameter> AddParameter(std::shared_ptr<IParameter> param);
        IParameter* AddParameter(IParameter* param);

        template<class T> std::weak_ptr<ITypedParameter<T>> GetParameter(const std::string& name) const;
        template<class Sig> bool ConnectCallback(const std::string& callback_name, const std::string& slot_name, IMetaObject* slot_owner, bool force_queue = false);
        
    protected:
        void AddConnection(std::shared_ptr<Connection>& connection, ISignal* sig);
        void AddConnection(boost::signals2::connection& connection, ISignal* sig);

        void AddCallback(ICallback*, const std::string& name);
        void AddSignal(std::weak_ptr<ISignal> signal, const std::string& name);
        void AddSlot(ISlot* slot, const std::string& name);
        void SetParameterRoot(const std::string& root);

        Context*                                 _ctx;
        SignalManager*                           _sig_manager;
        struct impl;
        impl* _pimpl;
    };
}