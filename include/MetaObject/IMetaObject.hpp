#pragma once
#include <IObject.h>
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Parameters/Buffers/BufferFactory.hpp"
#include <memory>
#include <mutex>
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
    class RelayManager;
    class ISignal;
    class ICallback;
    class ISlot;
    class Connection;
	class TypeInfo;
    class IVariableManager;
    
    class IParameter;
    class InputParameter;
    template<class T> class ITypedParameter;
    template<class T> class ITypedInputParameter;

    struct ParameterInfo;
    struct SignalInfo;
    struct SlotInfo;
    struct CallbackInfo;
    /*
      The IMetaObject interface class defines interfaces for introspection and serialization
      A IMetaObject derivative should use the IMetaObject macros for defining parameters, signals, 
      and slots.
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
      - Slots with a return value can only have a 1 to 1 mapping, thus the connection of a signal
        to a slot with a return will only call the most recent slot that was connected to it.
    */
	

    class MO_EXPORTS IMetaObject: public IObject
    {
    public:
		static int  Connect(IMetaObject* sender, const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name);
		static bool Connect(IMetaObject* sender, const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name, const TypeInfo& signature);
        template<class T> 
        static bool Connect(IMetaObject* sender, const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name);

        IMetaObject();
        virtual ~IMetaObject();
        
        // Setup
        virtual void  SetContext(Context* ctx);
        Context*      GetContext() const;
        virtual int   SetupSignals(RelayManager* mgr);
        virtual int   SetupVariableManager(IVariableManager* mgr);
        virtual void  BindSlots(bool firstInit) = 0;
        virtual void  Init(bool firstInit);
        virtual void  InitParameters(bool firstInit) = 0;
		virtual int   InitSignals(bool firstInit) = 0;
		
        virtual void  Serialize(ISimpleSerializer *pSerializer);
		virtual void  SerializeConnections(ISimpleSerializer* pSerializer);
        virtual void  SerializeParameters(ISimpleSerializer* pSerializer);

        // ------- Introspection
        // Get vector of info objects for each corresponding introspection class
        // optional name parameter can be used to get information for corresponding exact matches
        virtual std::vector<ParameterInfo*> GetParameterInfo() const = 0;
        virtual std::vector<ParameterInfo*> GetParameterInfo(const std::string& name) const;

        virtual std::vector<SignalInfo*>    GetSignalInfo() const = 0;
        virtual std::vector<SignalInfo*>    GetSignalInfo(const std::string& name) const;
        
        virtual std::vector<SlotInfo*>      GetSlotInfo() const = 0;
        virtual std::vector<SlotInfo*>      GetSlotInfo(const std::string& name) const;

        // -------- Signals / slots
        // If this class emits a signal by the given name, then the input sig will be added to the list of signals
        // that will be called when the signal is emitted.
        virtual bool ConnectByName(const std::string& signal_name, ISlot* slot);
		virtual bool ConnectByName(const std::string& slot_name, ISignal* signal);

        // Be careful to only call this once for each mgr object
        // This will call GetSignal<>(name) on the input mgr object and add the obtained signal
        // To the list of signals that is called whenever sig_{name} is emitted
        virtual int  ConnectByName(const std::string& name, RelayManager* mgr);
		virtual int  ConnectByName(const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name);
		virtual bool ConnectByName(const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name, const TypeInfo& signature);


        // Be careful to only call once for each mgr object
        // This will call mgr->GetSignal<>() for each declared signal
        virtual int  ConnectAll(RelayManager* mgr);

        virtual std::vector<std::pair<ISignal*, std::string>>  GetSignals() const;
        virtual std::vector<ISignal*>                          GetSignals(const std::string& name) const;
        virtual std::vector<std::pair<ISignal*, std::string>>  GetSignals(const TypeInfo& type) const;
		virtual ISignal*                                       GetSignal(const std::string& name, const TypeInfo& type) const;

        virtual std::vector<std::pair<ISlot*, std::string>>    GetSlots() const;
        virtual std::vector<ISlot*>                            GetSlots(const std::string& name) const;
        virtual std::vector<std::pair<ISlot*, std::string>>    GetSlots(const TypeInfo& signature) const;
        virtual ISlot*                                         GetSlot(const std::string& name, const TypeInfo& signature) const;
        template<class T> ISlot*                               GetSlot(const std::string& name) const;
    
        virtual int  DisconnectByName(const std::string& name);
        virtual bool Disconnect(ISignal* sig);
        virtual int  Disconnect(IMetaObject* obj);

        // Parameters
        virtual std::vector<IParameter*> GetDisplayParameters() const;
        
        std::vector<InputParameter*>                   GetInputs(const std::string& name_filter = "") const;
        std::vector<InputParameter*>                   GetInputs(const TypeInfo& type_filter, const std::string& name_filter = "") const;
        template<class T> std::vector<InputParameter*> GetInputs(const std::string& name_filter = "") const;

        InputParameter*                                GetInput(const std::string& name) const;
        template<class T> ITypedInputParameter<T>*     GetInput(const std::string& name);

        std::vector<IParameter*>                       GetOutputs(const std::string& name_filter = "") const;
        std::vector<IParameter*>                       GetOutputs(const TypeInfo& type_filter, const std::string& name_filter = "") const;
        template<class T> std::vector<IParameter*>     GetOutputs(const std::string& name_filter = "") const;

        IParameter*                                    GetOutput(const std::string& name) const;
        template<class T> ITypedParameter<T>*          GetOutput(const std::string& name);


        IParameter*              GetParameter(const std::string& name) const;
        IParameter*              GetParameterOptional(const std::string& name) const;
        std::vector<IParameter*> GetParameters(const std::string& filter = "") const;
        std::vector<IParameter*> GetParameters(const TypeInfo& filter) const;

        template<class T> T                   GetParameterValue(const std::string& name, long long ts = -1, Context* ctx = nullptr) const;
        template<class T> ITypedParameter<T>* GetParameter(const std::string& name) const;
        template<class T> ITypedParameter<T>* GetParameterOptional(const std::string& name) const;
        
        // Connects an input parameter to an output parameter
        bool ConnectInput(const std::string& input_name, IParameter* output, ParameterTypeFlags type = StreamBuffer_e);
        bool ConnectInput(InputParameter* input, IParameter* output, ParameterTypeFlags type = StreamBuffer_e);
        
    protected:
		friend class RelayManager;
		
        virtual IParameter* AddParameter(std::shared_ptr<IParameter> param);
        virtual IParameter* AddParameter(IParameter* param);

        template<class T> ITypedParameter<T>* UpdateParameter(const std::string& name, T& value, long long ts = -1, Context* ctx = nullptr);
        template<class T> ITypedParameter<T>* UpdateParameterPtr(const std::string& name, T& ptr);

        void AddSignal(ISignal* signal, const std::string& name);
        void AddSlot(ISlot* slot, const std::string& name);
        void SetParameterRoot(const std::string& root);
		void AddConnection(std::shared_ptr<Connection>& connection, const std::string& signal_name, const std::string& slot_name, const TypeInfo& signature, IMetaObject* obj = nullptr);

        struct	impl;

        impl*			_pimpl;
		Context*        _ctx;
		RelayManager*  _sig_manager;
        std::recursive_mutex _mtx;
    };
}