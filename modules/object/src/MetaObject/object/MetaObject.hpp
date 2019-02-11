#pragma once
#include "IMetaObject.hpp"
#include "MetaObject/object/detail/MetaObjectMacros.hpp"

namespace mo
{
    class MO_EXPORTS MetaObject : virtual public IMetaObject
    {
      public:
        using ParentClass = ct::VariadicTypedef<MetaObject>;
        MetaObject();
        virtual ~MetaObject() override;

        // Setup
        virtual void setStream(const IAsyncStreamPtr_t& ctx, bool overwrite = false) override;
        virtual IAsyncStreamPtr_t getStream() override;
        virtual int setupSignals(RelayManager* mgr) override;
        RelayManager* getRelayManager();
        virtual int setupParamServer(IParamServer* mgr) override;
        virtual int removeParamServer(IParamServer* mgr) override;

        virtual void Init(bool firstInit) override; // inherited from RCC, thus the PascalCase
        virtual void initCustom(bool firstInit) override;
        // virtual void initParams(bool firstInit) = 0;
        // virtual void bindSlots(bool firstInit) = 0;
        // virtual int initSignals(bool firstInit) = 0;
        // virtual void initOutputs() override = 0;

        virtual void Serialize(ISimpleSerializer* pSerializer) override; // Inherit from RCC's IObject
        virtual void serializeConnections(ISimpleSerializer* pSerializer) override;

        // ------- Introspection
        // Get vector of info objects for each corresponding introspection class
        // optional name Param can be used to get information for corresponding exact matches
        virtual void getParamInfo(ParamInfoVec_t& info) const override = 0;
        virtual ParamInfoVec_t getParamInfo(const std::string& name_filter) const override;
        virtual ParamInfoVec_t getParamInfo() const override;

        virtual void getSignalInfo(SignalInfoVec_t& info) const override = 0;
        virtual SignalInfoVec_t getSignalInfo(const std::string& name_filter) const override;
        virtual SignalInfoVec_t getSignalInfo() const override;

        virtual void getSlotInfo(SlotInfoVec_t& info) const override = 0;
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
        virtual int connectByName(const std::string& name, RelayManager* mgr) override;
        virtual int
        connectByName(const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name) override;
        virtual bool connectByName(const std::string& signal_name,
                                   IMetaObject* receiver,
                                   const std::string& slot_name,
                                   const TypeInfo& signature) override;

        // Be careful to only call once for each mgr object
        // This will call mgr->getSignal<>() for each declared signal
        virtual int connectAll(RelayManager* mgr) override;

        virtual std::vector<std::pair<ISignal*, std::string>> getSignals() const override;
        virtual std::vector<ISignal*> getSignals(const std::string& name) const override;
        virtual std::vector<std::pair<ISignal*, std::string>> getSignals(const TypeInfo& type) const override;
        virtual ISignal* getSignal(const std::string& name, const TypeInfo& type) const override;

        virtual std::vector<std::pair<ISlot*, std::string>> getSlots() const override;
        virtual std::vector<ISlot*> getSlots(const std::string& name) const override;
        virtual std::vector<std::pair<ISlot*, std::string>> getSlots(const TypeInfo& signature) const override;
        virtual ISlot* getSlot(const std::string& name, const TypeInfo& signature) const override;
        virtual UpdateSlot_t* getSlot_param_updated() const override;

        virtual int disconnectByName(const std::string& name) override;
        virtual bool disconnect(ISignal* sig) override;
        virtual int disconnect(IMetaObject* obj) override;

        // Params
        virtual ParamVec_t getDisplayParams() const override;

        virtual InputParamVec_t getInputs(const std::string& name_filter = "") const override;
        virtual InputParamVec_t getInputs(const TypeInfo& type_filter,
                                          const std::string& name_filter = "") const override;
        template <class T>
        InputParamVec_t getInputs(const std::string& name_filter = "") const;

        virtual InputParam* getInput(const std::string& name) const override;

        virtual ParamVec_t getOutputs(const std::string& name_filter = "") const override;
        virtual ParamVec_t getOutputs(const TypeInfo& type_filter, const std::string& name_filter = "") const override;

        virtual IParam* getOutput(const std::string& name) const override;

        virtual IParam* getParam(const std::string& name) const override;
        virtual IParam* getParamOptional(const std::string& name) const override;
        virtual ParamVec_t getParams(const std::string& filter = "") const override;
        virtual ParamVec_t getParams(const TypeInfo& filter) const override;
        virtual std::vector<IParamPtr_t> getImplicTParams() const override;

        // Connects an input Param to an output Param
        virtual bool connectInput(const std::string& input_name,
                                  IMetaObject* output_object,
                                  IParam* output_param,
                                  BufferFlags type = BLOCKING_STREAM_BUFFER) override;

        virtual bool connectInput(InputParam* input,
                                  IMetaObject* output_object,
                                  IParam* output_param,
                                  BufferFlags type = BLOCKING_STREAM_BUFFER) override;

        virtual Mutex_t& getMutex() const override;

        template <class T>
        ITInputParam<T>* getInput(const std::string& name);

        template <class T>
        TParam<T>* getOutput(const std::string& name) const;

        template <class T>
        T getParamValue(const std::string& name,
                        const OptionalTime& ts = OptionalTime(),
                        IAsyncStream* ctx = nullptr) const;

        template <class T>
        TParam<T>* getParam(const std::string& name) const;
        template <class T>
        TParam<T>* getParamOptional(const std::string& name) const;

      protected:
        IParam* addParam(IParamPtr_t param) override;
        IParam* addParam(IParam* param) override;

        void addSignal(ISignal* signal, const std::string& name) override;
        void addSlot(ISlot* slot, const std::string& name) override;
        void addSlot(std::unique_ptr<ISlot>&& slot, const std::string& name) override;
        void setParamRoot(const std::string& root) override;
        void addConnection(ConnectionPtr_t&& Connection,
                           const std::string& signal_name,
                           const std::string& slot_name,
                           const TypeInfo& signature,
                           IMetaObject* obj = nullptr) override;

        void onParamUpdate(IParam*, Header, UpdateFlags) override;

        template <class T>
        TParam<T>* updateParam(const std::string& name,
                               T& value,
                               const OptionalTime& ts = OptionalTime(),
                               IAsyncStream* ctx = nullptr);
        template <class T>
        TParam<T>* updateParam(const std::string& name,
                               const T& value,
                               const OptionalTime& ts = OptionalTime(),
                               IAsyncStream* ctx = nullptr);
        template <class T>
        TParam<T>* updateParamPtr(const std::string& name, T& ptr);

        friend class RelayManager;
        struct impl;

        std::unique_ptr<impl> _pimpl;
        IAsyncStreamPtr_t _ctx;
        RelayManager* _sig_manager;
    };
}
