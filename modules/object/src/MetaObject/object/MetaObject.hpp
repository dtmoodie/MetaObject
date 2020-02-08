#pragma once
#include "IMetaObject.hpp"
#include "MetaObject/object/detail/MetaObjectMacros.hpp"
#include <MetaObject/thread/Mutex.hpp>

#include <list>

namespace mo
{
    class MO_EXPORTS MetaObject : virtual public IMetaObject
    {
      public:
        using ParentClass = ct::VariadicTypedef<IMetaObject>;

        MO_BASE(MetaObject)
        PROPERTY(stream, &IMetaObject::getStream, &IMetaObject::setStream)
        MO_END;

        MetaObject();
        ~MetaObject() override;

        // Setup
        void setStream(const IAsyncStreamPtr_t& ctx);
        IAsyncStreamPtr_t getStream() const override;
        int setupSignals(const std::shared_ptr<RelayManager>& mgr) override;
        std::shared_ptr<RelayManager> getRelayManager() override;
        int setupParamServer(const std::shared_ptr<IParamServer>& mgr) override;
        int removeParamServer(IParamServer* mgr) override;

        void Init(bool firstInit) override; // inherited from RCC, thus the PascalCase
        void initCustom(bool firstInit) override;

        void Serialize(ISimpleSerializer* pSerializer) override; // Inherit from RCC's IObject
        void serializeConnections(ISimpleSerializer* pSerializer) override;

        // ------- Introspection
        // Get vector of info objects for each corresponding introspection class
        // optional name Param can be used to get information for corresponding exact matches
        void getParamInfo(ParamInfoVec_t& info) const override = 0;
        ParamInfoVec_t getParamInfo(const std::string& name_filter) const override;
        ParamInfoVec_t getParamInfo() const override;

        void getSignalInfo(SignalInfoVec_t& info) const override = 0;
        SignalInfoVec_t getSignalInfo(const std::string& name_filter) const override;
        SignalInfoVec_t getSignalInfo() const override;

        void getSlotInfo(SlotInfoVec_t& info) const override = 0;
        SlotInfoVec_t getSlotInfo(const std::string& name_filter) const override;
        SlotInfoVec_t getSlotInfo() const override;

        // -------- Signals / slots
        // If this class emits a signal by the given name, then the input sig will be added to the list of signals
        // that will be called when the signal is emitted.
        bool connectByName(const std::string& signal_name, ISlot* slot) override;
        bool connectByName(const std::string& slot_name, ISignal* signal) override;

        // Be careful to only call this once for each mgr object
        // This will call getSignal<>(name) on the input mgr object and add the obtained signal
        // To the list of signals that is called whenever sig_{name} is emitted
        int connectByName(const std::string& name, RelayManager* mgr) override;
        int connectByName(const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name) override;
        bool connectByName(const std::string& signal_name,
                           IMetaObject* receiver,
                           const std::string& slot_name,
                           const TypeInfo& signature) override;

        // Be careful to only call once for each mgr object
        // This will call mgr->getSignal<>() for each declared signal
        int connectAll(RelayManager* mgr) override;

        std::vector<std::pair<ISignal*, std::string>> getSignals() override;
        std::vector<ISignal*> getSignals(const std::string& name) override;
        std::vector<std::pair<ISignal*, std::string>> getSignals(const TypeInfo& type) override;
        ISignal* getSignal(const std::string& name, const TypeInfo& type) override;

        std::vector<std::pair<ISlot*, std::string>> getSlots() override;
        std::vector<ISlot*> getSlots(const std::string& name) override;
        std::vector<std::pair<ISlot*, std::string>> getSlots(const TypeInfo& signature) override;
        ISlot* getSlot(const std::string& name, const TypeInfo& signature) override;
        UpdateSlot_t* getSlot_param_updated() override;

        int disconnectByName(const std::string& name) override;
        bool disconnect(ISignal* sig) override;
        int disconnect(IMetaObject* obj) override;

        // Params
        ParamVec_t getDisplayParams() const override;

        InputParamVec_t getInputs(const std::string& name_filter = "") const override;
        InputParamVec_t getInputs(const TypeInfo& type_filter, const std::string& name_filter = "") const override;
        template <class T>
        InputParamVec_t getInputs(const std::string& name_filter = "") const;

        InputParam* getInput(const std::string& name) const override;

        ParamVec_t getOutputs(const std::string& name_filter = "") const override;
        ParamVec_t getOutputs(const TypeInfo& type_filter, const std::string& name_filter = "") const override;

        IParam* getOutput(const std::string& name) const override;

        IParam* getParam(const std::string& name) const override;
        IParam* getParamOptional(const std::string& name) const override;
        ParamVec_t getParams(const std::string& filter = "") const override;
        ParamVec_t getParams(const TypeInfo& filter) const override;
        std::vector<IParamPtr_t> getImplictParams() const override;

        // Connects an input Param to an output Param
        bool connectInput(const std::string& input_name,
                          IMetaObject* output_object,
                          const std::string& output_name,
                          BufferFlags type = BufferFlags::BLOCKING_STREAM_BUFFER) override;

        bool connectInput(InputParam* input,
                          IMetaObject* output_object,
                          IParam* output_param,
                          BufferFlags type = BufferFlags::BLOCKING_STREAM_BUFFER) override;

        Mutex_t& getMutex() const override;

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
                           rcc::shared_ptr<IMetaObject> obj = {}) override;

        void onParamUpdate(IParam*, Header, UpdateFlags) override;

        template <class T>
        TParam<T>* updateParam(const std::string& name, T& value, const OptionalTime& ts = OptionalTime());
        template <class T>
        TParam<T>* updateParam(const std::string& name, const T& value, const OptionalTime& ts = OptionalTime());
        template <class T>
        TParam<T>* updateParamPtr(const std::string& name, T& ptr);

        friend class RelayManager;
        bool isInitialized() const override;

      private:
        struct MO_EXPORTS ConnectionInfo
        {
            std::string signal_name;
            std::string slot_name;
            TypeInfo signature;
            rcc::weak_ptr<IMetaObject> obj;
            std::shared_ptr<Connection> connection;
        };

        struct MO_EXPORTS ParamConnectionInfo
        {
            ParamConnectionInfo() = default;
            ParamConnectionInfo(rcc::weak_ptr<IMetaObject> out,
                                std::string out_name,
                                std::string in_name,
                                BufferFlags type);
            rcc::weak_ptr<IMetaObject> output_object;
            std::string output_param;
            std::string input_param;
            BufferFlags connection_type;
        };

        void reconnectInputs();
        void reconnectSlots();
        void setupParamCallbacks();

        IAsyncStreamPtr_t m_stream;
        std::shared_ptr<RelayManager> m_sig_manager;

        std::map<std::string, std::map<TypeInfo, ISignal*>> m_signals;
        std::map<std::string, std::map<TypeInfo, ISlot*>> m_slots;
        std::vector<std::unique_ptr<ISlot>> m_implicit_slots;

        std::map<std::string, IParam*> m_params; // statically defined in object

        std::map<std::string, std::shared_ptr<IParam>> m_implicit_params; // Can be changed at runtime
        std::list<ConnectionInfo> m_connections;
        std::list<ParamConnectionInfo> m_param_connections;

        TSignal<void(IMetaObject*, Header, IParam*)> m_sig_param_updated;
        TSignal<void(IMetaObject*, IParam*)> m_sig_param_added;
        std::map<std::string, InputParam*> m_input_params;
        TSlot<Update_s> m_slot_param_updated;
        std::shared_ptr<IParamServer> m_param_server;
        mutable Mutex_t m_mutex;
        bool m_initialized = false;
    };
} // namespace mo

namespace std
{
    MO_EXPORTS ostream& operator<<(std::ostream& os, const std::pair<mo::ISignal*, std::string>& pair);

    MO_EXPORTS ostream& operator<<(std::ostream& os, const std::vector<std::pair<mo::ISignal*, std::string>>& signals);

    MO_EXPORTS ostream& operator<<(std::ostream& os, const std::pair<mo::ISlot*, std::string>& pair);

    MO_EXPORTS ostream& operator<<(std::ostream& os, const std::vector<std::pair<mo::ISlot*, std::string>>& slots);
} // namespace std