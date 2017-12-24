#pragma once
#include "MetaObject/core/detail/Enums.hpp"
#include "MetaObject/core/detail/Forward.hpp"
#include "MetaObject/core/detail/Time.hpp"
#include "MetaObject/detail/Export.hpp"
#include <RuntimeObjectSystem/IObject.h>
#include <RuntimeObjectSystem/shared_ptr.hpp>
#include <memory>

#ifndef MetaObject_EXPORTS
    #include "RuntimeObjectSystem/RuntimeLinkLibrary.h"
    #ifdef WIN32
        #pragma comment(lib, "Advapi32.lib")
        #ifdef _DEBUG
            RUNTIME_COMPILER_LINKLIBRARY("metaobject_objectd.lib")
        #else
            RUNTIME_COMPILER_LINKLIBRARY("metaobject_object.lib")
        #endif
    #else // Unix
        #ifdef NDEBUG
            RUNTIME_COMPILER_LINKLIBRARY("-lmetaobject_object")
        #else
            RUNTIME_COMPILER_LINKLIBRARY("-lmetaobject_objectd")
        #endif // NDEBUG
    #endif // WIN32
#endif // MetaObject_EXPORTS

#define MO_OBJ_TOOLTIP(tooltip) static std::string getTooltipStatic(){ return tooltip; }
#define MO_OBJ_DESCRIPTION(desc) static std::string getDescriptionStatic(){ return desc; }

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
    class MO_EXPORTS IMetaObject : public IObject
    {
    public:
        typedef IMetaObject Interface;
        typedef IMetaObjectInfo InterfaceInfo;
        typedef rcc::shared_ptr<IMetaObject> Ptr;
        typedef rcc::shared_ptr<const IMetaObject> ConstPtr;
        static const InterfaceID s_interfaceID = ct::ctcrc32("IMetaObject");
        static bool InheritsFrom(InterfaceID iid);

        static int  connect(IMetaObject* sender, const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name);
        static bool connect(IMetaObject* sender, const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name, const TypeInfo& signature);

        template <class T>
        static bool connect(IMetaObject* sender, const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name);

        virtual ~IMetaObject();

        // Setup
        virtual void setContext(const ContextPtr_t& ctx, bool overwrite = false) = 0;
        virtual ContextPtr_t getContext() = 0;
        virtual int setupSignals(RelayManager* mgr) = 0;
        virtual int setupVariableManager(IVariableManager* mgr) = 0;
        virtual int removeVariableManager(IVariableManager* mgr) = 0;
        virtual void bindSlots(bool firstInit) = 0;
        virtual void Init(bool firstInit) = 0;      // inherited from RCC, thus the PascalCase
        virtual void initCustom(bool firstInit) = 0;
        virtual void initParams(bool firstInit) = 0;
        virtual int initSignals(bool firstInit) = 0;
        virtual void initOutputs() = 0;

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
        virtual int  connectByName(const std::string& name, RelayManager* mgr) = 0;
        virtual int  connectByName(const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name) = 0;
        virtual bool connectByName(const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name, const TypeInfo& signature) = 0;

        // Be careful to only call once for each mgr object
        // This will call mgr->getSignal<>() for each declared signal
        virtual int connectAll(RelayManager* mgr) = 0;

        virtual std::vector<std::pair<ISignal*, std::string> > getSignals() const = 0;
        virtual std::vector<ISignal*> getSignals(const std::string& name) const = 0;
        virtual std::vector<std::pair<ISignal*, std::string> > getSignals(const TypeInfo& type) const = 0;
        virtual ISignal* getSignal(const std::string& name, const TypeInfo& type) const = 0;

        virtual std::vector<std::pair<ISlot*, std::string> > getSlots() const = 0;
        virtual std::vector<ISlot*> getSlots(const std::string& name) const = 0;
        virtual std::vector<std::pair<ISlot*, std::string> > getSlots(const TypeInfo& signature) const = 0;
        virtual ISlot* getSlot(const std::string& name, const TypeInfo& signature) const = 0;
        virtual TSlot<void(IParam*, Context*, OptionalTime_t, size_t, const std::shared_ptr<ICoordinateSystem>&, UpdateFlags)>* getSlot_param_updated() const = 0;

        virtual int disconnectByName(const std::string& name) = 0;
        virtual bool disconnect(ISignal* sig) = 0;
        virtual int disconnect(IMetaObject* obj) = 0;

        // Params
        virtual ParamVec_t getDisplayParams() const = 0;

        virtual std::vector<InputParam*> getInputs(const std::string& name_filter = "") const = 0;
        virtual std::vector<InputParam*> getInputs(const TypeInfo& type_filter, const std::string& name_filter = "") const = 0;
        template <class T>
        std::vector<InputParam*> getInputs(const std::string& name_filter = "") const;

        virtual InputParam* getInput(const std::string& name) const = 0;

        virtual ParamVec_t getOutputs(const std::string& name_filter = "") const = 0;
        virtual ParamVec_t getOutputs(const TypeInfo& type_filter, const std::string& name_filter = "") const = 0;

        virtual IParam* getOutput(const std::string& name) const = 0;

        virtual IParam* getParam(const std::string& name) const = 0;
        virtual IParam* getParamOptional(const std::string& name) const = 0;
        virtual std::vector<IParam*> getParams(const std::string& filter = "") const = 0;
        virtual std::vector<IParam*> getParams(const TypeInfo& filter) const = 0;
        virtual std::vector<std::shared_ptr<IParam>> getImplicitParams() const = 0;

        // Connects an input Param to an output Param
        virtual bool connectInput(const std::string& input_name, IMetaObject* output_object,
                                  IParam* output_param, ParamType type = StreamBuffer_e) = 0;
        virtual bool connectInput(InputParam* input,
                                  IMetaObject* output_object, IParam* output_param,
                                  ParamType type = StreamBuffer_e) = 0;
        static bool  connectInput(IMetaObject* output_object, IParam*     output_Param,
                                  IMetaObject* input_object,  InputParam* input_param,
                                  ParamType type = StreamBuffer_e);

        virtual Mutex_t& getMutex() const = 0;

    protected:
        friend class RelayManager;

        virtual IParam* addParam(std::shared_ptr<IParam> param) = 0;
        virtual IParam* addParam(IParam* param) = 0;

        virtual void addSignal(ISignal* signal, const std::string& name) = 0;
        virtual void addSlot(ISlot* slot, const std::string& name) = 0;
        virtual void setParamRoot(const std::string& root) = 0;
        virtual void addConnection(std::shared_ptr<Connection>&& Connection,
                                   const std::string& signal_name,
                                   const std::string& slot_name,
                                   const TypeInfo& signature,
                                   IMetaObject* obj = nullptr) = 0;

        virtual void onParamUpdate(IParam*, Context*, OptionalTime_t, size_t, const std::shared_ptr<ICoordinateSystem>&, UpdateFlags) = 0;
    };


    class MO_EXPORTS MetaObject: public IMetaObject
    {
    public:
        MetaObject();
        virtual ~MetaObject();

        // Setup
        virtual void setContext(const ContextPtr_t& ctx, bool overwrite = false) override;
        virtual ContextPtr_t getContext() override;
        virtual int setupSignals(RelayManager* mgr) override;
        virtual int setupVariableManager(IVariableManager* mgr) override;
        virtual int removeVariableManager(IVariableManager* mgr) override;
        virtual void bindSlots(bool firstInit) = 0;
        virtual void Init(bool firstInit) override;      // inherited from RCC, thus the PascalCase
        virtual void initCustom(bool firstInit) override;
        virtual void initParams(bool firstInit) = 0;
        virtual int  initSignals(bool firstInit) = 0;
        virtual void initOutputs() = 0;

        virtual void Serialize(ISimpleSerializer* pSerializer) override; // Inherit from RCC's IObject
        virtual void serializeConnections(ISimpleSerializer* pSerializer) override;
        virtual void serializeParams(ISimpleSerializer* pSerializer) override;

        // ------- Introspection
        // Get vector of info objects for each corresponding introspection class
        // optional name Param can be used to get information for corresponding exact matches
        virtual void           getParamInfo(ParamInfoVec_t& info) const = 0;
        virtual ParamInfoVec_t getParamInfo(const std::string& name_filter) const override;
        virtual ParamInfoVec_t getParamInfo() const override;

        virtual void            getSignalInfo(SignalInfoVec_t& info) const = 0;
        virtual SignalInfoVec_t getSignalInfo(const std::string& name_filter) const override;
        virtual SignalInfoVec_t getSignalInfo() const override;

        virtual void          getSlotInfo(SlotInfoVec_t& info) const = 0;
        virtual SlotInfoVec_t getSlotInfo(const std::string& name_filter) const override;
        virtual SlotInfoVec_t getSlotInfo() const override;

        // -------- Signals / slots
        // If this class emits a signal by the given name, then the input sig will be added to the list of signals
        // that will be called when the signal is emitted.
        virtual bool connectByName(const std::string& signal_name, ISlot* slot) override;
        virtual bool connectByName(const std::string& slot_name, ISignal* signal) override;

        // Be careful to only call this once for each mgr object
        // This will call getSignal<>(name) on the input mgr object and add the obtained signal
        // To the list of signals that is called whenever sig_{name} is emitted
        virtual int  connectByName(const std::string& name, RelayManager* mgr) override;
        virtual int  connectByName(const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name) override;
        virtual bool connectByName(const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name, const TypeInfo& signature) override;

        // Be careful to only call once for each mgr object
        // This will call mgr->getSignal<>() for each declared signal
        virtual int connectAll(RelayManager* mgr) override;

        virtual std::vector<std::pair<ISignal*, std::string> > getSignals() const override;
        virtual std::vector<ISignal*> getSignals(const std::string& name) const override;
        virtual std::vector<std::pair<ISignal*, std::string> > getSignals(const TypeInfo& type) const override;
        virtual ISignal* getSignal(const std::string& name, const TypeInfo& type) const override;

        virtual std::vector<std::pair<ISlot*, std::string> > getSlots() const override;
        virtual std::vector<ISlot*> getSlots(const std::string& name) const override;
        virtual std::vector<std::pair<ISlot*, std::string> > getSlots(const TypeInfo& signature) const override;
        virtual ISlot* getSlot(const std::string& name, const TypeInfo& signature) const override;
        TSlot<void(IParam*, Context*, OptionalTime_t, size_t, const std::shared_ptr<ICoordinateSystem>&, UpdateFlags)>* getSlot_param_updated() const override;

        virtual int  disconnectByName(const std::string& name) override;
        virtual bool disconnect(ISignal* sig) override;
        virtual int  disconnect(IMetaObject* obj) override;

        // Params
        virtual ParamVec_t getDisplayParams() const override;

        virtual std::vector<InputParam*> getInputs(const std::string& name_filter = "") const override;
        virtual std::vector<InputParam*> getInputs(const TypeInfo& type_filter, const std::string& name_filter = "") const override;
        template <class T>
        std::vector<InputParam*> getInputs(const std::string& name_filter = "") const;

        virtual InputParam* getInput(const std::string& name) const override;

        virtual ParamVec_t getOutputs(const std::string& name_filter = "") const override;
        virtual ParamVec_t getOutputs(const TypeInfo& type_filter, const std::string& name_filter = "") const override;

        virtual IParam* getOutput(const std::string& name) const override;

        virtual IParam* getParam(const std::string& name) const override;
        virtual IParam* getParamOptional(const std::string& name) const override;
        virtual std::vector<IParam*> getParams(const std::string& filter = "") const override;
        virtual std::vector<IParam*> getParams(const TypeInfo& filter) const override;
        virtual std::vector<std::shared_ptr<IParam>> getImplicitParams() const override;

        // Connects an input Param to an output Param
        virtual bool connectInput(const std::string& input_name, IMetaObject* output_object, IParam* output_param, ParamType type = StreamBuffer_e) override;
        virtual bool connectInput(InputParam* input, IMetaObject* output_object, IParam* output_param, ParamType type = StreamBuffer_e) override;
        virtual Mutex_t& getMutex() const override;

        template <class T>
        ITInputParam<T>* getInput(const std::string& name);

        template <class T>
        ITParam<T>* getOutput(const std::string& name) const;

        template <class T>
        T getParamValue(const std::string& name, const OptionalTime_t& ts = OptionalTime_t(), Context* ctx = nullptr) const;
        template <class T>
        ITParam<T>* getParam(const std::string& name) const;
        template <class T>
        ITParam<T>* getParamOptional(const std::string& name) const;
        template <class T>
        TSlot<T>* getSlot(const std::string& name) const;


    protected:
        virtual IParam* addParam(std::shared_ptr<IParam> param) override;
        virtual IParam* addParam(IParam* param) override;

        virtual void addSignal(ISignal* signal, const std::string& name) override;
        virtual void addSlot(ISlot* slot, const std::string& name) override;
        virtual void setParamRoot(const std::string& root) override;
        virtual void addConnection(std::shared_ptr<Connection>&& Connection,
                                   const std::string& signal_name,
                                   const std::string& slot_name,
                                   const TypeInfo& signature,
                                   IMetaObject* obj = nullptr) override;

        virtual void onParamUpdate(IParam*, Context*, OptionalTime_t, size_t, const std::shared_ptr<ICoordinateSystem>&, UpdateFlags) override;


        template <class T>
        ITParam<T>* updateParam(const std::string& name, T& value, const OptionalTime_t& ts = OptionalTime_t(), Context* ctx = nullptr);
        template <class T>
        ITParam<T>* updateParam(const std::string& name, const T& value, const OptionalTime_t& ts = OptionalTime_t(), Context* ctx = nullptr);
        template <class T>
        ITParam<T>* updateParamPtr(const std::string& name, T& ptr);


        friend class RelayManager;
        struct impl;

        impl* _pimpl;
        ContextPtr_t _ctx;
        RelayManager* _sig_manager;
    };

}
