#pragma once
#include "MetaObject/core/detail/Enums.hpp"
#include "MetaObject/core/detail/Forward.hpp"
#include "MetaObject/core/detail/Time.hpp"
#include "MetaObject/detail/Export.hpp"
#include <RuntimeObjectSystem/IObject.h>
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

namespace mo {
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
class MO_EXPORTS IMetaObject : public IObject {
public:
    typedef IMetaObject Interface;
    typedef IMetaObjectInfo InterfaceInfo;
    typedef rcc::shared_ptr<IMetaObject> Ptr;
    typedef rcc::shared_ptr<const IMetaObject> ConstPtr;

    static int connect(IMetaObject* sender, const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name);
    static bool connect(IMetaObject* sender, const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name, const TypeInfo& signature);
    template <class T>
    static bool connect(IMetaObject* sender, const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name);

    IMetaObject();
    virtual ~IMetaObject();

    // Setup
    virtual void setContext(const ContextPtr_t& ctx, bool overwrite = false);
    virtual ContextPtr_t getContext();
    virtual int setupSignals(RelayManager* mgr);
    virtual int setupVariableManager(IVariableManager* mgr);
    virtual int removeVariableManager(IVariableManager* mgr);
    virtual void bindSlots(bool firstInit) = 0;
    virtual void Init(bool firstInit); // inherited from RCC
    virtual void initCustom(bool firstInit);
    virtual void initParams(bool firstInit) = 0;
    virtual int initSignals(bool firstInit) = 0;
    virtual void initOutputs() = 0;

    virtual void Serialize(ISimpleSerializer* pSerializer); // Inherit from RCC's IObject
    virtual void serializeConnections(ISimpleSerializer* pSerializer);
    virtual void serializeParams(ISimpleSerializer* pSerializer);

    // ------- Introspection
    // Get vector of info objects for each corresponding introspection class
    // optional name Param can be used to get information for corresponding exact matches
    virtual void getParamInfo(ParamInfoVec_t& info) const = 0;
    ParamInfoVec_t getParamInfo(const std::string& name_filter) const;
    ParamInfoVec_t getParamInfo() const;

    virtual void getSignalInfo(SignalInfoVec_t& info) const = 0;
    SignalInfoVec_t getSignalInfo(const std::string& name_filter) const;
    SignalInfoVec_t getSignalInfo() const;

    virtual void getSlotInfo(SlotInfoVec_t& info) const = 0;
    SlotInfoVec_t getSlotInfo(const std::string& name_filter) const;
    SlotInfoVec_t getSlotInfo() const;

    // -------- Signals / slots
    // If this class emits a signal by the given name, then the input sig will be added to the list of signals
    // that will be called when the signal is emitted.
    virtual bool connectByName(const std::string& signal_name, ISlot* slot);
    virtual bool connectByName(const std::string& slot_name, ISignal* signal);

    // Be careful to only call this once for each mgr object
    // This will call getSignal<>(name) on the input mgr object and add the obtained signal
    // To the list of signals that is called whenever sig_{name} is emitted
    virtual int connectByName(const std::string& name, RelayManager* mgr);
    virtual int connectByName(const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name);
    virtual bool connectByName(const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name, const TypeInfo& signature);

    // Be careful to only call once for each mgr object
    // This will call mgr->getSignal<>() for each declared signal
    virtual int connectAll(RelayManager* mgr);

    virtual std::vector<std::pair<ISignal*, std::string> > getSignals() const;
    virtual std::vector<ISignal*> getSignals(const std::string& name) const;
    virtual std::vector<std::pair<ISignal*, std::string> > getSignals(const TypeInfo& type) const;
    virtual ISignal* getSignal(const std::string& name, const TypeInfo& type) const;

    virtual std::vector<std::pair<ISlot*, std::string> > getSlots() const;
    virtual std::vector<ISlot*> getSlots(const std::string& name) const;
    virtual std::vector<std::pair<ISlot*, std::string> > getSlots(const TypeInfo& signature) const;
    virtual ISlot* getSlot(const std::string& name, const TypeInfo& signature) const;
    TSlot<void(IParam*, Context*, OptionalTime_t, size_t, const std::shared_ptr<ICoordinateSystem>&, UpdateFlags)>* getSlot_param_updated() const;
    template <class T>
    TSlot<T>* getSlot(const std::string& name) const;

    virtual int disconnectByName(const std::string& name);
    virtual bool disconnect(ISignal* sig);
    virtual int disconnect(IMetaObject* obj);

    // Params
    virtual ParamVec_t getDisplayParams() const;

    virtual std::vector<InputParam*> getInputs(const std::string& name_filter = "") const;
    virtual std::vector<InputParam*> getInputs(const TypeInfo& type_filter, const std::string& name_filter = "") const;
    template <class T>
    std::vector<InputParam*> getInputs(const std::string& name_filter = "") const;

    virtual InputParam* getInput(const std::string& name) const;
    template <class T>
    ITInputParam<T>* getInput(const std::string& name);

    virtual ParamVec_t getOutputs(const std::string& name_filter = "") const;
    virtual ParamVec_t getOutputs(const TypeInfo& type_filter, const std::string& name_filter = "") const;
    //template<class T> std::vector<ITParam<T>*> getOutputs(const std::string& name_filter = "") const;

    virtual IParam* getOutput(const std::string& name) const;
    template <class T>
    ITParam<T>* getOutput(const std::string& name) const;

    virtual IParam* getParam(const std::string& name) const;
    virtual IParam* getParamOptional(const std::string& name) const;
    virtual std::vector<IParam*> getParams(const std::string& filter = "") const;
    virtual std::vector<IParam*> getParams(const TypeInfo& filter) const;
    virtual std::vector<std::shared_ptr<IParam>> getImplicitParams() const;

    template <class T>
    T getParamValue(const std::string& name, const OptionalTime_t& ts = OptionalTime_t(), Context* ctx = nullptr) const;
    template <class T>
    ITParam<T>* getParam(const std::string& name) const;
    template <class T>
    ITParam<T>* getParamOptional(const std::string& name) const;

    // Connects an input Param to an output Param
    bool connectInput(const std::string& input_name, IMetaObject* output_object, IParam* output_param, ParamType type = StreamBuffer_e);
    bool connectInput(InputParam* input, IMetaObject* output_object, IParam* output_param, ParamType type = StreamBuffer_e);
    static bool connectInput(IMetaObject* output_object, IParam* output_Param,
        IMetaObject* input_object, InputParam* input_param, ParamType type = StreamBuffer_e);

    Mutex_t& getMutex();

protected:
    virtual IParam* addParam(std::shared_ptr<IParam> param);
    virtual IParam* addParam(IParam* param);

    template <class T>
    ITParam<T>* updateParam(const std::string& name, T& value, const OptionalTime_t& ts = OptionalTime_t(), Context* ctx = nullptr);
    template <class T>
    ITParam<T>* updateParam(const std::string& name, const T& value, const OptionalTime_t& ts = OptionalTime_t(), Context* ctx = nullptr);
    template <class T>
    ITParam<T>* updateParamPtr(const std::string& name, T& ptr);

    void addSignal(ISignal* signal, const std::string& name);
    void addSlot(ISlot* slot, const std::string& name);
    void setParamRoot(const std::string& root);
    void addConnection(std::shared_ptr<Connection>&& Connection, const std::string& signal_name, const std::string& slot_name, const TypeInfo& signature, IMetaObject* obj = nullptr);
    virtual void onParamUpdate(IParam*, Context*, OptionalTime_t, size_t, const std::shared_ptr<ICoordinateSystem>&, UpdateFlags);

    friend class RelayManager;
    struct impl;

    impl* _pimpl;
    ContextPtr_t _ctx;
    RelayManager* _sig_manager;
    Mutex_t* _mtx;
};
}
