#include "MetaObject/object/IMetaObject.hpp"

#include "MetaObject/core/TypeTable.hpp"
#include "MetaObject/logging/logging.hpp"
#include "MetaObject/object/RelayManager.hpp"
#include "MetaObject/object/detail/IMetaObjectImpl.hpp"
#include "MetaObject/object/detail/IMetaObject_pImpl.hpp"
#include "MetaObject/params/IParam.hpp"
#include "MetaObject/params/InputParam.hpp"
#include "MetaObject/params/ParamServer.hpp"
#include "MetaObject/params/buffers/BufferFactory.hpp"
#include "MetaObject/params/buffers/IBuffer.hpp"
#include "MetaObject/signals/ISignal.hpp"
#include "MetaObject/signals/ISlot.hpp"
#include "MetaObject/signals/SignalInfo.hpp"
#include "MetaObject/signals/SlotInfo.hpp"

#include "RuntimeObjectSystem/IObjectState.hpp"
#include "RuntimeObjectSystem/ISimpleSerializer.h"

#include <MetaObject/thread/fiber_include.hpp>

namespace std
{
    ostream& operator<<(std::ostream& os, const std::pair<mo::ISignal*, std::string>& pair)
    {
        os << "(" << pair.second << " [" << pair.first->getSignature().name() << "]";
        return os;
    }

    ostream& operator<<(std::ostream& os, const std::vector<std::pair<mo::ISignal*, std::string>>& signals)
    {
        for (const auto& sig : signals)
        {
            os << sig;
            os << std::endl;
        }

        return os;
    }

    ostream& operator<<(std::ostream& os, const std::pair<mo::ISlot*, std::string>& pair)
    {
        os << "(" << pair.second << " [" << pair.first->getSignature().name() << "]";
        return os;
    }

    ostream& operator<<(std::ostream& os, const std::vector<std::pair<mo::ISlot*, std::string>>& slots)
    {
        for (const auto& slot : slots)
        {
            os << slot;
            os << std::endl;
        }

        return os;
        return os;
    }
} // namespace std

namespace mo
{
    IMetaObject::IMetaObject() = default;
    IMetaObject::~IMetaObject() = default;

    int IMetaObject::connect(IMetaObject& sender,
                             const std::string& signal_name,
                             IMetaObject& receiver,
                             const std::string& slot_name)
    {
        int count = 0;
        auto my_signals = sender.getSignals(signal_name);
        auto my_slots = receiver.getSlots(slot_name);

        for (auto signal : my_signals)
        {
            for (auto slot : my_slots)
            {
                if (signal->getSignature() == slot->getSignature())
                {
                    auto connection = slot->connect(signal);
                    if (connection)
                    {
                        sender.addConnection(std::move(connection),
                                             signal_name,
                                             slot_name,
                                             slot->getSignature(),
                                             rcc::shared_ptr<IMetaObject>(receiver));
                        ++count;
                    }
                    break;
                }
                MO_LOG(debug,
                       "Signature mismatch, slot '{}' <{}> != signal '{}' <{}>",
                       slot_name,
                       slot->getSignature().name(),
                       signal_name,
                       signal->getSignature().name());
            }
        }

        return count;
    }

    bool IMetaObject::connect(IMetaObject& sender,
                              const std::string& signal_name,
                              IMetaObject& receiver,
                              const std::string& slot_name,
                              const TypeInfo& signature)
    {
        auto signal = sender.getSignal(signal_name, signature);
        if (!signal)
        {
            MO_LOG(info,
                   "Unable to find a signal with name {} and signature {} in object of type {}. Signals:\n{}",
                   signal_name,
                   signature.name(),
                   sender.GetTypeName(),
                   sender.getSignals());
            return false;
        }
        auto slot = receiver.getSlot(slot_name, signature);
        if (!slot)
        {
            MO_LOG(info,
                   "Unable to find a slot with name {} and signature {} in object of type {}, Slots: \n{}",
                   slot_name,
                   signature.name(),
                   receiver.GetTypeName(),
                   receiver.getSlots());
            return false;
        }
        auto connection = slot->connect(signal);
        if (!connection)
        {
            MO_LOG(info,
                   "Unable to connect signal '{}' to slot '{}' with signature {}",
                   signal_name,
                   slot_name,
                   signature.name());
            return false;
        }
        sender.addConnection(
            std::move(connection), signal_name, slot_name, signature, rcc::shared_ptr<IMetaObject>(receiver));
        return true;
    }

    MetaObject::MetaObject()
    {
        m_slot_param_updated = std::bind(
            &MetaObject::onParamUpdate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
        m_signals["param_updated"][m_sig_param_updated.getSignature()] = &m_sig_param_updated;
        m_signals["param_added"][m_sig_param_updated.getSignature()] = &m_sig_param_added;

        setStream(IAsyncStream::current());
    }

    MetaObject::~MetaObject()
    {
        if (m_param_server)
        {
            m_param_server->removeParam(this);
        }
    }

    Mutex_t& MetaObject::getMutex() const
    {
        return m_mutex;
    }

    void MetaObject::reconnectInputs()
    {
        auto connections_copy = m_param_connections;
        m_param_connections.clear();
        for (auto& param_connection : connections_copy)
        {
            rcc::shared_ptr<IMetaObject> obj = param_connection.output_object.lock();
            if (obj)
            {
                auto output = obj->getOutput(param_connection.output_param);
                if (output == nullptr)
                {
                    MO_LOG(debug,
                           "Unable to find {} in  reinitializing",
                           param_connection.output_param,
                           obj->GetTypeName());
                    obj->initParams(false);
                    output = obj->getOutput(param_connection.output_param);
                    if (output == nullptr)
                    {
                        MO_LOG(info,
                               "Unable to find {} in {} unable to reconnect {} from object {}",
                               param_connection.output_param,
                               obj->GetTypeName(),
                               param_connection.input_param,
                               this->GetTypeName());
                        continue;
                    }
                }
                auto input = this->getInput(param_connection.input_param);
                if (input)
                {
                    if (this->connectInput(input, obj.get(), output, param_connection.connection_type))
                    {
                        input->getData();
                        MO_LOG(debug,
                               "Reconnected {}:{} to {}:{}",
                               GetTypeName(),
                               param_connection.input_param,
                               obj->GetTypeName(),
                               param_connection.output_param);
                    }
                    else
                    {
                        MO_LOG(info,
                               "Reconnect failed {}:{} to {}:{}",
                               GetTypeName(),
                               param_connection.input_param,
                               obj->GetTypeName(),
                               param_connection.output_param);
                    }
                }
                else
                {
                    MO_LOG(debug,
                           "Unable to find input param {} in object {}",
                           param_connection.input_param,
                           this->GetTypeName());
                }
            }
            else
            {
                MO_LOG(debug,
                       "Output object no longer exists for input [{}] expexted output named '{}'",
                       param_connection.input_param,
                       param_connection.output_param);
            }
        }
    }

    void MetaObject::reconnectSlots()
    {
        for (auto& connection : m_connections)
        {
            auto obj = connection.obj.lock();
            if (obj)
            {
                auto signal = this->getSignal(connection.signal_name, connection.signature);
                auto slot = obj->getSlot(connection.slot_name, connection.signature);
                if (signal == nullptr)
                {
                    MO_LOG(info,
                           "Unable to find signal with name '{}' and signature <{}> in new object of type {}. "
                           "Signals:\n{}",
                           connection.signal_name,
                           connection.signature.name(),
                           this->GetTypeName(),
                           this->getSignals());
                }
                if (slot == nullptr)
                {
                    // This function is only called when we are doing a recompile, thus we know that first init == false
                    auto obj = connection.obj.lock();
                    if (obj)
                    {
                        if (!obj->isInitialized())
                        {
                            obj->Init(false);
                        }
                        slot = obj->getSlot(connection.slot_name, connection.signature);
                        if (slot == nullptr)
                        {
                            MO_LOG(info,
                                   "Unable to find slot with name '{}' and signature <{}> in new object of type {}. "
                                   "Slots:\n{}",
                                   connection.slot_name,
                                   connection.signature.name(),
                                   obj->GetTypeName(),
                                   obj->getSlots());
                        }
                    }
                    else
                    {
                        MO_LOG(info, "Object no longer exists");
                    }
                }
                if (signal && slot)
                {
                    auto connection_ = slot->connect(signal);
                    if (connection_)
                    {
                        connection.connection = connection_;
                    }
                    else
                    {
                        MO_LOG(info,
                               "Unable to reconnect signal '{}' <{}> to slot '{}' <{}>",
                               connection.signal_name,
                               signal->getSignature().name(),
                               connection.slot_name,
                               slot->getSignature().name());
                    }
                }
            }
        }
    }

    void MetaObject::setupParamCallbacks()
    {
        auto params = getParams();
        for (auto param : params)
        {
            auto update_slot = IMetaObject::template getSlot<Update_s>("on_" + param->getName() + "_modified");
            if (update_slot)
            {
                auto connection = param->registerUpdateNotifier(update_slot);
                if (connection)
                {
                    this->addConnection(std::move(connection),
                                        param->getName() + "_modified",
                                        "on_" + param->getName() + "_modified",
                                        update_slot->getSignature(),
                                        rcc::shared_ptr<IMetaObject>(*this));
                }
            }
            auto delete_slot =
                IMetaObject::template getSlot<void(IParam const*)>("on_" + param->getName() + "_deleted");
            if (delete_slot)
            {
                auto connection = param->registerDeleteNotifier(delete_slot);
                if (connection)
                {
                    this->addConnection(std::move(connection),
                                        param->getName() + "_deleted",
                                        "on_" + param->getName() + "_modified",
                                        update_slot->getSignature(),
                                        rcc::shared_ptr<IMetaObject>(*this));
                }
            }
        }
    }

    void MetaObject::Init(bool first_init)
    {
        if (m_initialized)
        {
            return;
        }
        initParams(first_init);
        initSignals(first_init);
        bindSlots(first_init);
        initCustom(first_init);
        setupParamCallbacks();

        if (!first_init)
        {
            reconnectInputs();
            reconnectSlots();
        }
        m_initialized = true;
    }

    void MetaObject::initCustom(bool /*firstInit*/)
    {
    }

    int MetaObject::setupSignals(const std::shared_ptr<RelayManager>& manager)
    {
        m_sig_manager = manager;
        int count = 0;
        for (auto& my_slots : m_slots)
        {
            for (auto& slot : my_slots.second)
            {
                ConnectionInfo info;
                info.connection = manager->connect(slot.second, my_slots.first, this);
                info.slot_name = my_slots.first;
                info.signature = slot.first;
                m_connections.push_back(info);
                ++count;
            }
        }

        for (auto& my_signals : m_signals)
        {
            for (auto& signal : my_signals.second)
            {
                auto Connection = manager->connect(signal.second, my_signals.first, this);
                ConnectionInfo info;
                info.signal_name = my_signals.first;
                info.signature = signal.first;
                info.connection = Connection;
                m_connections.push_back(info);
                ++count;
            }
        }

        return count;
    }

    std::shared_ptr<RelayManager> MetaObject::getRelayManager()
    {
        return m_sig_manager;
    }

    int MetaObject::setupParamServer(const std::shared_ptr<IParamServer>& server)
    {
        if (m_param_server != nullptr)
        {
            removeParamServer(m_param_server.get());
        }
        m_param_server = server;
        int count = 0;
        for (auto& param : m_implicit_params)
        {
            server->addParam(this, param.second.get());
            ++count;
        }
        for (auto& param : m_params)
        {
            server->addParam(this, param.second);
            ++count;
        }
        return count;
    }

    int MetaObject::removeParamServer(IParamServer* mgr)
    {
        int count = 0;
        mgr->removeParam(this);
        return count;
    }

    void MetaObject::Serialize(ISimpleSerializer* pSerializer)
    {
        IObject::Serialize(pSerializer);
        serializeConnections(pSerializer);
        serializeParams(pSerializer);
    }

    void MetaObject::serializeConnections(ISimpleSerializer* pSerializer)
    {
        SERIALIZE(m_connections);
        SERIALIZE(m_param_connections);
        SERIALIZE(m_stream);
        SERIALIZE(m_sig_manager);
        SERIALIZE(m_param_server);
    }

    void MetaObject::setStream(const IAsyncStreamPtr_t& ctx)
    {
        if (ctx == nullptr && m_stream != nullptr)
        {
            MO_LOG(info, "Setting stream to nullptr");
        }
        m_stream = ctx;
        auto ptr = ctx.get();
        for (auto& param : m_implicit_params)
        {
            param.second->setStream(ptr);
        }
        for (auto& param : m_params)
        {
            param.second->setStream(ptr);
        }
    }

    IAsyncStreamPtr_t MetaObject::getStream() const
    {
        return m_stream;
    }

    int MetaObject::disconnectByName(const std::string& name)
    {
        auto my_signals = this->getSignals(name);
        int count = 0;
        for (auto& sig : my_signals)
        {
            count += sig->disconnect() ? 1 : 0;
        }
        return count;
    }

    bool MetaObject::disconnect(ISignal*)
    {
        return false;
    }

    int MetaObject::disconnect(IMetaObject* obj)
    {
        auto obj_signals = obj->getSignals();
        int count = 0;
        for (auto signal : obj_signals)
        {
            count += disconnect(signal.first) ? 1 : 0;
        }
        return count;
    }

    std::vector<IParam*> MetaObject::getDisplayParams() const
    {
        std::vector<IParam*> output;
        for (auto& param : m_params)
        {
            output.push_back(param.second);
        }
        for (auto& param : m_implicit_params)
        {
            output.push_back(param.second.get());
        }
        return output;
    }

    IParam* MetaObject::getParam(const std::string& name) const
    {
        auto param = this->getParamOptional(name);
        if (!param)
        {
            THROW(debug, "Param with name {} not found", name);
        }
        return param;
    }

    std::vector<IParam*> MetaObject::getParams(const std::string& filter) const
    {
        std::vector<IParam*> output;
        for (auto& itr : m_params)
        {
            if (!filter.empty())
            {
                if (itr.first.find(filter) != std::string::npos)
                {
                    output.push_back(itr.second);
                }
            }
            else
            {
                output.push_back(itr.second);
            }
        }
        for (auto& itr : m_implicit_params)
        {
            if (!filter.empty())
            {
                if (itr.first.find(filter) != std::string::npos)
                {
                    output.push_back(itr.second.get());
                }
            }
            else
            {
                output.push_back(itr.second.get());
            }
        }
        return output;
    }

    std::vector<IParam*> MetaObject::getParams(const TypeInfo& filter) const
    {
        std::vector<IParam*> output;
        for (auto& itr : m_params)
        {
            if (itr.second->getTypeInfo() == filter)
            {
                output.push_back(itr.second);
            }
        }
        for (auto& itr : m_implicit_params)
        {
            if (itr.second->getTypeInfo() == filter)
            {
                output.push_back(itr.second.get());
            }
        }
        return output;
    }

    std::vector<std::shared_ptr<IParam>> MetaObject::getImplictParams() const
    {
        std::vector<std::shared_ptr<IParam>> output;
        for (const auto& param : m_implicit_params)
        {
            output.emplace_back(param.second);
        }
        return output;
    }

    IParam* MetaObject::getParamOptional(const std::string& name) const
    {
        auto itr = m_params.find(name);
        if (itr != m_params.end())
        {
            return itr->second;
        }
        auto itr2 = m_implicit_params.find(name);
        if (itr2 != m_implicit_params.end())
        {
            return itr2->second.get();
        }
        MO_LOG(trace, "Param with name {} not found", name);
        return nullptr;
    }

    InputParam* MetaObject::getInput(const std::string& name) const
    {
        auto itr = m_input_params.find(name);
        if (itr != m_input_params.end())
        {
            return itr->second;
        }
        return nullptr;
    }

    std::vector<InputParam*> MetaObject::getInputs(const std::string& name_filter) const
    {
        return getInputs(TypeInfo::Void(), name_filter);
    }

    std::vector<InputParam*> MetaObject::getInputs(const TypeInfo& type_filter, const std::string& name_filter) const
    {
        std::vector<InputParam*> output;
        for (auto param : m_params)
        {
            if (param.second->checkFlags(ParamFlags::kINPUT))
            {
                if (param.second->getTypeInfo() == type_filter || type_filter == TypeInfo::Void())
                {
                    if (!name_filter.empty())
                    {
                        if (name_filter.find(param.first) != std::string::npos)
                        {
                            if (auto out = dynamic_cast<InputParam*>(param.second))
                            {
                                output.push_back(out);
                            }
                        }
                    }
                    else
                    {
                        if (auto out = dynamic_cast<InputParam*>(param.second))
                        {
                            output.push_back(out);
                        }
                    }
                }
            }
        }
        for (auto param : m_implicit_params)
        {
            if (param.second->checkFlags(ParamFlags::kINPUT))
            {
                if (param.second->getTypeInfo() == type_filter || type_filter == TypeInfo::Void())
                {
                    if (!name_filter.empty())
                    {
                        if (name_filter.find(param.first) != std::string::npos)
                        {
                            if (auto out = dynamic_cast<InputParam*>(param.second.get()))
                            {
                                output.push_back(out);
                            }
                        }
                    }
                    else
                    {
                        if (auto out = dynamic_cast<InputParam*>(param.second.get()))
                        {
                            output.push_back(out);
                        }
                    }
                }
            }
        }
        return output;
    }

    IParam* MetaObject::getOutput(const std::string& name) const
    {
        auto itr = m_params.find(name);
        if (itr != m_params.end())
        {
            if (itr->second->checkFlags(mo::ParamFlags::kOUTPUT))
            {
                return itr->second;
            }
        }
        auto itr2 = m_implicit_params.find(name);
        if (itr2 != m_implicit_params.end())
        {
            if (itr->second->checkFlags(mo::ParamFlags::kOUTPUT))
            {
                return itr2->second.get();
            }
        }
        // do a pass through all params to see if the param was renamed
        for (const auto& itr : m_params)
        {
            if (itr.second->getName() == name)
            {
                if (itr.second->checkFlags(mo::ParamFlags::kOUTPUT))
                {
                    return itr.second;
                }
            }
        }
        for (const auto& itr : m_implicit_params)
        {
            if (itr.second->getName() == name)
            {
                if (itr.second->checkFlags(mo::ParamFlags::kOUTPUT))
                {
                    return itr.second.get();
                }
            }
        }
        return nullptr;
    }

    std::vector<IParam*> MetaObject::getOutputs(const std::string& name_filter) const
    {
        return getOutputs(TypeInfo::Void(), name_filter);
    }

    std::vector<IParam*> MetaObject::getOutputs(const TypeInfo& type_filter, const std::string& name_filter) const
    {
        std::vector<IParam*> output;
        for (auto param : m_params)
        {
            if (param.second->checkFlags(ParamFlags::kOUTPUT))
            {
                if (!name_filter.empty())
                {
                    if (name_filter.find(param.first) != std::string::npos)
                    {
                        if (param.second->getTypeInfo() == type_filter || type_filter == TypeInfo::Void())
                        {
                            output.push_back(param.second);
                        }
                    }
                }
                else
                {
                    if (param.second->getTypeInfo() == type_filter || type_filter == TypeInfo::Void())
                    {
                        output.push_back(param.second);
                    }
                }
            }
        }
        for (auto param : m_implicit_params)
        {
            if (param.second->checkFlags(ParamFlags::kOUTPUT))
            {
                if (!name_filter.empty())
                {
                    if (name_filter.find(param.first) != std::string::npos)
                    {
                        if (param.second->getTypeInfo() == type_filter || type_filter == TypeInfo::Void())
                            output.push_back(param.second.get());
                    }
                }
                else
                {
                    if (param.second->getTypeInfo() == type_filter || type_filter == TypeInfo::Void())
                        output.push_back(param.second.get());
                }
            }
        }
        return output;
    }

    bool MetaObject::connectInput(const std::string& input_name,
                                  IMetaObject* output_object,
                                  const std::string& output_name,
                                  BufferFlags type_)
    {
        auto input = getInput(input_name);
        auto output = output_object->getOutput(output_name);
        if (input && output)
        {
            return connectInput(input, output_object, output, type_);
        }

        auto inputs = getInputs();
        auto print_inputs = [inputs]() -> std::string {
            std::stringstream ss;
            for (auto _input : inputs)
            {
                ss << dynamic_cast<IParam*>(_input)->getName() << ", ";
            }
            return ss.str();
        };
        MO_LOG(debug,
               "Unable to find input with name {} in object {} with inputs {}",
               input_name,
               this->GetTypeName(),
               print_inputs());
        return false;
    }

    bool MetaObject::connectInput(InputParam* input, IMetaObject* output_object, IParam* output, BufferFlags type_)
    {
        if (input == nullptr || output == nullptr)
        {
            MO_LOG(debug, "NULL input or output passed in");
            return false;
        }

        if (input && input->acceptsInput(output))
        {
            // Check contexts to see if a buffer needs to be setup
            auto output_stream = output->getStream();
            auto input_stream = input->getStream();
            if (type_ == BufferFlags::DEFAULT && output_stream != input_stream)
            {
                type_ = getDefaultBufferType(output_stream, input_stream);
            }
            const bool force = type_ & BufferFlags::FORCE_BUFFERED;
            const bool buffer_input = input->checkFlags(mo::ParamFlags::kREQUIRE_BUFFERED);
            const bool buffer_output = output->checkFlags(mo::ParamFlags::kREQUIRE_BUFFERED);
            if (force || buffer_input || buffer_output)
            {
                type_ = BufferFlags(type_ & ~ct::value(BufferFlags::FORCE_BUFFERED));
                auto buffer = buffer::BufferFactory::instance()->createBuffer(output, type_);
                if (!buffer)
                {
                    MO_LOG(warn,
                           "Unable to create {} for datatype {}",
                           bufferFlagsToString(type_),
                           TypeTable::instance()->typeToName(output->getTypeInfo()));
                    return false;
                }
                std::string buffer_type = bufferFlagsToString(type_);
                buffer->setName(output->getTreeName() + " " + buffer_type + " buffer for " + input->getTreeName());
                if (input->setInput(buffer))
                {
                    if (output_object)
                    {
                        m_param_connections.emplace_back(
                            rcc::weak_ptr<IMetaObject>(*output_object), output->getName(), input->getName(), type_);
                    }

                    return true;
                }

                MO_LOG(debug,
                       "Failed to connect output {} [{}] to input {} [{}]",
                       output->getName(),
                       TypeTable::instance()->typeToName(output->getTypeInfo()),
                       dynamic_cast<IParam*>(input)->getName(),
                       TypeTable::instance()->typeToName(dynamic_cast<IParam*>(input)->getTypeInfo()));
                return false;
            }
            if (output_stream && m_stream.get())
            {
                if (output_stream->threadId() != m_stream.get()->threadId())
                {
                    auto buffer = buffer::BufferFactory::instance()->createBuffer(output, type_);
                    if (buffer)
                    {
                        buffer->setName(output->getTreeName() + " buffer for " + input->getTreeName());
                        if (input->setInput(buffer))
                        {
                            if (output_object)
                            {
                                m_param_connections.emplace_back(rcc::weak_ptr<IMetaObject>(*output_object),
                                                                 output->getName(),
                                                                 input->getName(),
                                                                 type_);
                            }

                            return true;
                        }

                        MO_LOG(debug,
                               "Failed to connect output {} [{}] to input {} [{}]",
                               output->getName(),
                               TypeTable::instance()->typeToName(output->getTypeInfo()),
                               dynamic_cast<IParam*>(input)->getName(),
                               TypeTable::instance()->typeToName(dynamic_cast<IParam*>(input)->getTypeInfo()));
                        return false;
                    }

                    MO_LOG(debug,
                           "No buffer of desired type found for type {}",
                           TypeTable::instance()->typeToName(output->getTypeInfo()));
                }
                else
                {
                    if (input->setInput(output))
                    {
                        if (output_object)
                        {
                            m_param_connections.emplace_back(
                                rcc::weak_ptr<IMetaObject>(*output_object), output->getName(), input->getName(), type_);
                        }

                        return true;
                    }
                    MO_LOG(debug,
                           "Failed to connect output {} [{}] to input {} [{}]",
                           output->getName(),
                           TypeTable::instance()->typeToName(output->getTypeInfo()),
                           dynamic_cast<IParam*>(input)->getName(),
                           TypeTable::instance()->typeToName(dynamic_cast<IParam*>(input)->getTypeInfo()));
                    return false;
                }
            }
            else
            {
                if (input->setInput(output))
                {
                    if (output_object)
                    {
                        m_param_connections.emplace_back(
                            rcc::weak_ptr<IMetaObject>(*output_object), output->getName(), input->getName(), type_);
                    }

                    return true;
                }

                MO_LOG(debug,
                       "Failed to connect output {} [{}] to input {} [{}]",
                       output->getName(),
                       TypeTable::instance()->typeToName(output->getTypeInfo()),
                       dynamic_cast<IParam*>(input)->getName(),
                       TypeTable::instance()->typeToName(dynamic_cast<IParam*>(input)->getTypeInfo()));
                return false;
            }
        }
        else
        {
            MO_LOG(debug,
                   "Input {} does not accept input of type {}",
                   input->getTreeName(),
                   TypeTable::instance()->typeToName(output->getTypeInfo()));
        }
        return false;
    }

    bool IMetaObject::connectInput(
        IMetaObject* out_obj, IParam* out_param, IMetaObject* in_obj, InputParam* in_param, BufferFlags type)
    {
        return in_obj->connectInput(in_param, out_obj, out_param, type);
    }

    IParam* MetaObject::addParam(std::shared_ptr<IParam> param)
    {
        param->setMtx(&getMutex());
        param->setStream(m_stream.get());
#ifdef _DEBUG
        for (auto& param_ : m_params)
        {
            if (param_.second == param.get())
            {
                MO_LOG(debug, "Trying to add a param a second time");
                return param.get();
            }
        }
#endif
        m_implicit_params[param->getName()] = param;
        if (param->checkFlags(ParamFlags::kINPUT))
        {
            m_input_params[param->getName()] = dynamic_cast<InputParam*>(param.get());
        }
        auto connection = param->registerUpdateNotifier(&(this->m_slot_param_updated));
        m_sig_param_added(this, param.get());
        return param.get();
    }

    IParam* MetaObject::addParam(IParam* param)
    {
        param->setMtx(&getMutex());
        param->setStream(m_stream.get());
#ifdef _DEBUG
        for (auto& param_ : m_params)
        {
            if (param_.second == param)
            {
                MO_LOG(debug, "Trying to add a param a second time");
                return param;
            }
        }
#endif
        m_params[param->getName()] = param;
        if (param->checkFlags(ParamFlags::kINPUT))
        {
            m_input_params[param->getName()] = dynamic_cast<InputParam*>(param);
        }
        auto connection = param->registerUpdateNotifier(&(this->m_slot_param_updated));
        m_sig_param_added(this, param);
        this->addConnection(std::move(connection),
                            "param_updated",
                            "param_updated",
                            this->m_slot_param_updated.getSignature(),
                            rcc::shared_ptr<IMetaObject>(*this));
        return param;
    }

    void MetaObject::setParamRoot(const std::string& root)
    {
        for (auto& param : m_params)
        {
            param.second->setTreeRoot(root);
        }
        for (auto& param : m_implicit_params)
        {
            param.second->setTreeRoot(root);
        }
    }

    std::vector<ParamInfo*> MetaObject::getParamInfo(const std::string& /*name*/) const
    {
        std::vector<ParamInfo*> output;
        getParamInfo(output);

        return output;
    }

    std::vector<ParamInfo*> MetaObject::getParamInfo() const
    {
        std::vector<ParamInfo*> output;
        getParamInfo(output);

        return output;
    }

    std::vector<SignalInfo*> MetaObject::getSignalInfo(const std::string&) const
    {
        std::vector<SignalInfo*> info;
        getSignalInfo(info);

        return info;
    }

    std::vector<SignalInfo*> MetaObject::getSignalInfo() const
    {
        std::vector<SignalInfo*> info;
        getSignalInfo(info);
        return info;
    }

    std::vector<SlotInfo*> MetaObject::getSlotInfo() const
    {
        std::vector<SlotInfo*> output;
        getSlotInfo(output);
        return output;
    }

    std::vector<SlotInfo*> MetaObject::getSlotInfo(const std::string& name) const
    {
        std::vector<SlotInfo*> tmp;
        getSlotInfo(tmp);
        std::vector<SlotInfo*> output;
        for (auto& itr : tmp)
        {
            if (itr->name.find(name) != std::string::npos)
                output.push_back(itr);
        }

        return output;
    }

    std::vector<std::pair<ISlot*, std::string>> MetaObject::getSlots()
    {
        std::vector<std::pair<ISlot*, std::string>> my_slots;
        for (auto itr1 : m_slots)
        {
            for (auto itr2 : itr1.second)
            {
                my_slots.emplace_back(std::make_pair(itr2.second, itr1.first));
            }
        }
        return my_slots;
    }

    std::vector<ISlot*> MetaObject::getSlots(const std::string& name)
    {
        std::vector<ISlot*> output;
        auto itr = m_slots.find(name);
        if (itr != m_slots.end())
        {
            for (auto slot : itr->second)
            {
                output.push_back(slot.second);
            }
        }
        return output;
    }

    std::vector<std::pair<ISlot*, std::string>> MetaObject::getSlots(const TypeInfo& signature)
    {
        std::vector<std::pair<ISlot*, std::string>> output;
        for (auto& type : m_slots)
        {
            auto itr = type.second.find(signature);
            if (itr != type.second.end())
            {
                output.emplace_back(std::make_pair(itr->second, type.first));
            }
        }
        return output;
    }

    ISlot* MetaObject::getSlot(const std::string& name, const TypeInfo& signature)
    {
        auto itr1 = m_slots.find(name);
        if (itr1 != m_slots.end())
        {
            auto itr2 = itr1->second.find(signature);
            if (itr2 != itr1->second.end())
            {
                return itr2->second;
            }
        }
        if (name == "param_updated" && signature == m_slot_param_updated.getSignature())
        {
            return &m_slot_param_updated;
        }
        return nullptr;
    }

    TSlot<Update_s>* MetaObject::getSlot_param_updated()
    {
        return &m_slot_param_updated;
    }

    bool MetaObject::connectByName(const std::string& name, ISlot* slot)
    {
        auto signal = getSignal(name, slot->getSignature());
        if (signal)
        {
            auto connection = signal->connect(slot);
            if (connection)
            {
                addConnection(std::move(connection), name, "", slot->getSignature());
                return true;
            }
        }
        return false;
    }

    bool MetaObject::connectByName(const std::string& name, ISignal* signal)
    {
        auto slot = getSlot(name, signal->getSignature());
        if (slot)
        {
            auto connection = slot->connect(signal);
            if (connection)
            {
                addConnection(std::move(connection), "", name, signal->getSignature());
                return true;
            }
        }
        return false;
    }

    int MetaObject::connectByName(const std::string& /*name*/, RelayManager* /*mgr*/)
    {
        // TODO implement
        return 0;
    }

    int MetaObject::connectByName(const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name)
    {
        int count = 0;
        auto my_signals = getSignals(signal_name);
        auto my_slots = receiver->getSlots(slot_name);
        for (auto signal : my_signals)
        {
            for (auto slot : my_slots)
            {
                if (signal->getSignature() == slot->getSignature())
                {
                    auto connection = slot->connect(signal);
                    if (connection)
                    {
                        addConnection(std::move(connection),
                                      signal_name,
                                      slot_name,
                                      slot->getSignature(),
                                      rcc::shared_ptr<IMetaObject>(*receiver));
                        ++count;
                        break;
                    }
                }
            }
        }
        return count;
    }

    bool MetaObject::connectByName(const std::string& signal_name,
                                   IMetaObject* receiver,
                                   const std::string& slot_name,
                                   const TypeInfo& signature)
    {
        auto signal = getSignal(signal_name, signature);
        auto slot = receiver->getSlot(slot_name, signature);
        if (signal && slot)
        {
            auto connection = slot->connect(signal);
            if (connection)
            {
                addConnection(
                    std::move(connection), signal_name, slot_name, signature, rcc::shared_ptr<IMetaObject>(*receiver));
                return true;
            }
        }
        return false;
    }

    int MetaObject::connectAll(RelayManager* mgr)
    {
        auto my_signals = getSignalInfo();
        int count = 0;
        for (auto& signal : my_signals)
        {
            count += connectByName(signal->name, mgr);
        }
        return count;
    }

    void MetaObject::addSlot(ISlot* slot, const std::string& name)
    {
        m_slots[name][slot->getSignature()] = slot;
    }

    void MetaObject::addSlot(std::unique_ptr<ISlot>&& slot, const std::string& name)
    {
        addSlot(slot.get(), name);
        m_implicit_slots.emplace_back(std::move(slot));
    }

    void MetaObject::addSignal(ISignal* sig, const std::string& name)
    {
        m_signals[name][sig->getSignature()] = sig;
        if (m_sig_manager)
        {
            auto connection = m_sig_manager->connect(sig, name, this);
            ConnectionInfo info;
            info.signal_name = name;
            info.signature = sig->getSignature();
            info.connection = connection;
            m_connections.push_back(info);
        }
    }

    std::vector<std::pair<ISignal*, std::string>> MetaObject::getSignals()
    {
        std::vector<std::pair<ISignal*, std::string>> my_signals;
        for (auto& name_itr : m_signals)
        {
            for (auto& sig_itr : name_itr.second)
            {
                my_signals.emplace_back(std::make_pair(sig_itr.second, name_itr.first));
            }
        }
        return my_signals;
    }

    std::vector<ISignal*> MetaObject::getSignals(const std::string& name)
    {
        std::vector<ISignal*> my_signals;
        auto itr = m_signals.find(name);
        if (itr != m_signals.end())
        {
            for (auto& sig_itr : itr->second)
            {
                my_signals.push_back(sig_itr.second);
            }
        }
        return my_signals;
    }

    std::vector<std::pair<ISignal*, std::string>> MetaObject::getSignals(const TypeInfo& type)
    {
        std::vector<std::pair<ISignal*, std::string>> my_signals;
        for (auto& name_itr : m_signals)
        {
            auto type_itr = name_itr.second.find(type);
            if (type_itr != name_itr.second.end())
            {
                my_signals.emplace_back(std::make_pair(type_itr->second, name_itr.first));
            }
        }
        return my_signals;
    }

    ISignal* MetaObject::getSignal(const std::string& name, const TypeInfo& type)
    {
        auto name_itr = m_signals.find(name);
        if (name_itr != m_signals.end())
        {
            auto type_itr = name_itr->second.find(type);
            if (type_itr != name_itr->second.end())
            {
                return type_itr->second;
            }
        }
        if (name == "param_updated" && type == m_sig_param_updated.getSignature())
        {
            return &m_sig_param_updated;
        }
        if (name == "param_added" && type == m_sig_param_added.getSignature())
        {
            return &m_sig_param_added;
        }
        return nullptr;
    }

    void MetaObject::addConnection(std::shared_ptr<Connection>&& connection,
                                   const std::string& signal_name,
                                   const std::string& slot_name,
                                   const TypeInfo& signature,
                                   rcc::shared_ptr<IMetaObject> obj)
    {
        ConnectionInfo info;
        info.connection = std::move(connection);
        info.obj = rcc::weak_ptr<IMetaObject>(obj);
        info.signal_name = signal_name;
        info.slot_name = slot_name;
        info.signature = signature;
        m_connections.push_back(info);
    }

    void MetaObject::onParamUpdate(IParam* param, Header hdr, UpdateFlags)
    {
        m_sig_param_updated(this, hdr, param);
    }

    MetaObject::ParamConnectionInfo::ParamConnectionInfo(rcc::weak_ptr<mo::IMetaObject> out,
                                                         std::string out_name,
                                                         std::string in_name,
                                                         BufferFlags type)
        : output_object(std::move(out))
        , output_param(std::move(out_name))
        , input_param(std::move(in_name))
        , connection_type(type)
    {
    }

    bool MetaObject::isInitialized() const
    {
        return m_initialized;
    }

} // namespace mo
