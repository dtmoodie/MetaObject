#pragma once
#include "MetaObject/core/detail/Enums.hpp"
#include "MetaObject/core/detail/Forward.hpp"
#include "MetaObject/core/detail/Time.hpp"
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/object.hpp"

#include <RuntimeObjectSystem/IObject.h>
#include <RuntimeObjectSystem/shared_ptr.hpp>

#include <memory>
#include <string>

namespace mo
{
    /*
      The IMetaObject interface class defines interfaces for introspection and serialization
      A IMetaObject derivative should use the IMetaObject macros for defining Params, signals,
      and slots.
      Params - Outputs, Inputs, Control, and State.
       - Outputs Params are shared with other IMetaObjects
       - Inputs Params are read from other IMetaObjects
       - Control Params are user set settings
       - State Params are used status introspection
     Signals
      - functions that are called by an IMetaObject that invoke all Connected slots
      - must have void return type
      - must handle asynchronous operation
     Slots
      - functions that are called when a signal is invoked
      - must have void return type
      - should be called on the thread of the owning context
      - Slots with a return value can only have a 1 to 1 mapping, thus the Connection of a signal
        to a slot with a return will only call the most recent slot that was Connected to it.
    */
    template<class T>
    struct TMetaObjectInterfaceHelper;

    class MO_EXPORTS IMetaObject : virtual public TInterface<IMetaObject, IObject>
    {
      public:
        using ParentClass = ct::VariadicTypedef<IMetaObject>;
        using Interface = IMetaObject;
        using InterfaceInfo = IMetaObjectInfo;
        template<class T>
        using InterfaceHelper = TMetaObjectInterfaceHelper<T>;

        using Ptr = rcc::shared_ptr<IMetaObject>;
        using ConstPtr = rcc::shared_ptr<const IMetaObject>;

        static int connect(IMetaObject* sender,
                           const std::string& signal_name,
                           IMetaObject* receiver,
                           const std::string& slot_name);

        static bool connect(IMetaObject* sender,
                            const std::string& signal_name,
                            IMetaObject* receiver,
                            const std::string& slot_name,
                            const TypeInfo& signature);

        template <class T>
        static bool connect(IMetaObject* sender,
                            const std::string& signal_name,
                            IMetaObject* receiver,
                            const std::string& slot_name);

        virtual ~IMetaObject();

        // Setup
        virtual void setContext(const ContextPtr_t& ctx, bool overwrite = false) = 0;
        virtual ContextPtr_t getContext() = 0;
        virtual int setupSignals(RelayManager* mgr) = 0;
        virtual int setupVariableManager(IVariableManager* mgr) = 0;
        virtual int removeVariableManager(IVariableManager* mgr) = 0;

        virtual void Init(bool firstInit) = 0; // inherited from RCC, thus the PascalCase
        virtual void initCustom(bool firstInit) = 0;
        virtual void bindSlots(bool firstInit) = 0;
        virtual void initParams(bool firstInit) = 0;
        virtual int initSignals(bool firstInit) = 0;

        virtual void Serialize(ISimpleSerializer* pSerializer) = 0; // Inherit from RCC's IObject
        virtual void serializeConnections(ISimpleSerializer* pSerializer) = 0;
        virtual void serializeParams(ISimpleSerializer* pSerializer) = 0;

        // ------- Introspection
        // Get vector of info objects for each corresponding introspection class
        // optional name Param can be used to get information for corresponding exact matches
        virtual void getParamInfo(ParamInfoVec_t& info) const = 0;
        virtual ParamInfoVec_t getParamInfo(const std::string& name_filter) const = 0;
        virtual ParamInfoVec_t getParamInfo() const = 0;

        virtual void getSignalInfo(SignalInfoVec_t& info) const = 0;
        virtual SignalInfoVec_t getSignalInfo(const std::string& name_filter) const = 0;
        virtual SignalInfoVec_t getSignalInfo() const = 0;

        virtual void getSlotInfo(SlotInfoVec_t& info) const = 0;
        virtual SlotInfoVec_t getSlotInfo(const std::string& name_filter) const = 0;
        virtual SlotInfoVec_t getSlotInfo() const = 0;

        // -------- Signals / slots
        // If this class emits a signal by the given name, then the input sig will be added to the list of signals
        // that will be called when the signal is emitted.
        virtual bool connectByName(const std::string& signal_name, ISlot* slot) = 0;
        virtual bool connectByName(const std::string& slot_name, ISignal* signal) = 0;

        // Be careful to only call this once for each mgr object
        // This will call getSignal<>(name) on the input mgr object and add the obtained signal
        // To the list of signals that is called whenever sig_{name} is emitted
        virtual int connectByName(const std::string& name, RelayManager* mgr) = 0;
        virtual int
        connectByName(const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name) = 0;

        virtual bool connectByName(const std::string& signal_name,
                                   IMetaObject* receiver,
                                   const std::string& slot_name,
                                   const TypeInfo& signature) = 0;

        // Be careful to only call once for each mgr object
        // This will call mgr->getSignal<>() for each declared signal
        virtual int connectAll(RelayManager* mgr) = 0;

        virtual std::vector<std::pair<ISignal*, std::string>> getSignals() const = 0;
        virtual std::vector<ISignal*> getSignals(const std::string& name) const = 0;
        virtual std::vector<std::pair<ISignal*, std::string>> getSignals(const TypeInfo& type) const = 0;
        virtual ISignal* getSignal(const std::string& name, const TypeInfo& type) const = 0;

        virtual std::vector<std::pair<ISlot*, std::string>> getSlots() const = 0;
        virtual std::vector<ISlot*> getSlots(const std::string& name) const = 0;
        virtual std::vector<std::pair<ISlot*, std::string>> getSlots(const TypeInfo& signature) const = 0;
        virtual ISlot* getSlot(const std::string& name, const TypeInfo& signature) const = 0;
        virtual UpdateSlot_t* getSlot_param_updated() const = 0;

        virtual int disconnectByName(const std::string& name) = 0;
        virtual bool disconnect(ISignal* sig) = 0;
        virtual int disconnect(IMetaObject* obj) = 0;

        // Params
        virtual ParamVec_t getDisplayParams() const = 0;

        virtual InputParamVec_t getInputs(const std::string& name_filter = "") const = 0;
        virtual InputParamVec_t getInputs(const TypeInfo& type_filter, const std::string& name_filter = "") const = 0;
        template <class T>
        InputParamVec_t getInputs(const std::string& name_filter = "") const;

        virtual InputParam* getInput(const std::string& name) const = 0;

        virtual ParamVec_t getOutputs(const std::string& name_filter = "") const = 0;
        virtual ParamVec_t getOutputs(const TypeInfo& type_filter, const std::string& name_filter = "") const = 0;

        virtual IParam* getOutput(const std::string& name) const = 0;

        virtual IParam* getParam(const std::string& name) const = 0;
        virtual IParam* getParamOptional(const std::string& name) const = 0;
        virtual ParamVec_t getParams(const std::string& filter = "") const = 0;
        virtual ParamVec_t getParams(const TypeInfo& filter) const = 0;
        virtual std::vector<IParamPtr_t> getImplicitParams() const = 0;

        // Connects an input Param to an output Param
        virtual bool connectInput(const std::string& input_name,
                                  IMetaObject* output_object,
                                  IParam* output_param,
                                  BufferFlags type = Default_e) = 0;

        virtual bool connectInput(InputParam* input,
                                  IMetaObject* output_object,
                                  IParam* output_param,
                                  BufferFlags type = Default_e) = 0;

        static bool connectInput(IMetaObject* output_object,
                                 IParam* output_Param,
                                 IMetaObject* input_object,
                                 InputParam* input_param,
                                 BufferFlags type = Default_e);

        virtual Mutex_t& getMutex() const = 0;

      protected:
        friend class RelayManager;
        friend class MetaObject;

        virtual IParam* addParam(IParamPtr_t param) = 0;
        virtual IParam* addParam(IParam* param) = 0;

        virtual void addSignal(ISignal* signal, const std::string& name) = 0;
        virtual void addSlot(ISlot* slot, const std::string& name) = 0;
        virtual void setParamRoot(const std::string& root) = 0;
        virtual void addConnection(ConnectionPtr_t&& Connection,
                                   const std::string& signal_name,
                                   const std::string& slot_name,
                                   const TypeInfo& signature,
                                   IMetaObject* obj = nullptr) = 0;

        virtual void
        onParamUpdate(IParam*, Context*, OptionalTime_t, size_t, const CoordinateSystemPtr_t&, UpdateFlags) = 0;
    };

} // namespace mo
