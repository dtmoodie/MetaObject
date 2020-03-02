#include "TMultiInput-inl.hpp"
#include <MetaObject/signals/Connection.hpp>
#include <MetaObject/thread/Mutex.hpp>
#include <MetaObject/thread/fiber_include.hpp>

namespace mo
{

    IMultiInput::IMultiInput()
        : IParam()
        , InputParam()
    {
    }

    void IMultiInput::setInputs(const std::vector<InputParam*>& inputs)
    {
        m_inputs = inputs;
    }

    bool IMultiInput::setInput(const std::shared_ptr<IParam>& publisher)
    {
        Lock_t lock(mtx());
        for (auto input : m_inputs)
        {
            if (input->acceptsInput(publisher.get()))
            {
                if (input->setInput(publisher))
                {
                    if (m_current_input)
                    {
                        m_current_input->setInput(nullptr);
                    }
                    m_current_input = input;
                    return true;
                }
            }
        }
        return false;
    }

    bool IMultiInput::setInput(IParam* publisher)
    {
        Lock_t lock(mtx());
        for (auto input : m_inputs)
        {
            if (input->acceptsInput(publisher))
            {
                if (input->setInput(publisher))
                {
                    if (m_current_input != nullptr)
                    {
                        m_current_input->setInput(nullptr);
                    }
                    m_current_input = input;
                    return true;
                }
            }
        }
        return false;
    }

    IMultiInput::IContainerPtr_t IMultiInput::getData(const Header& desired)
    {
        Lock_t lock(mtx());
        if (m_current_input)
        {
            return m_current_input->getData(desired);
        }
        return {};
    }

    IMultiInput::IContainerConstPtr_t IMultiInput::getData(const Header& desired) const
    {
        Lock_t lock(mtx());
        if (m_current_input)
        {
            return m_current_input->getData(desired);
        }
        return {};
    }

    void IMultiInput::setMtx(Mutex_t* mtx)
    {
        for (auto input : m_inputs)
        {
            input->setMtx(mtx);
        }
    }

    mo::TypeInfo IMultiInput::getTypeInfo() const
    {
        Lock_t lock(mtx());
        if (m_current_input)
        {
            return m_current_input->getTypeInfo();
        }

        return _void_type_info;
    }

    const mo::TypeInfo IMultiInput::_void_type_info = mo::TypeInfo(typeid(void));

    mo::IParam* IMultiInput::getInputParam() const
    {
        if (m_current_input)
        {
            return m_current_input->getInputParam();
        }
        return nullptr;
    }

    OptionalTime IMultiInput::getInputTimestamp()
    {
        if (m_current_input)
        {
            return m_current_input->getInputTimestamp();
        }
        return {};
    }

    FrameNumber IMultiInput::getInputFrameNumber()
    {
        if (m_current_input)
        {
            return m_current_input->getInputFrameNumber();
        }
        return {};
    }

    OptionalTime IMultiInput::getTimestamp() const
    {
        if (m_current_input)
        {
            return m_current_input->getTimestamp();
        }
        return {};
    }

    FrameNumber IMultiInput::getFrameNumber() const
    {
        if (m_current_input)
        {
            return m_current_input->getFrameNumber();
        }
        return -1;
    }

    bool IMultiInput::isInputSet() const
    {
        return m_current_input != nullptr;
    }

    bool IMultiInput::acceptsInput(IParam* input) const
    {
        for (auto in : m_inputs)
        {
            if (in->acceptsInput(input))
            {
                return true;
            }
        }
        return false;
    }

    bool IMultiInput::acceptsType(const TypeInfo& type) const
    {
        for (auto in : m_inputs)
        {
            if (in->acceptsType(type))
            {
                return true;
            }
        }
        return false;
    }

    ConnectionPtr_t IMultiInput::registerUpdateNotifier(ISlot& f)
    {
        ConnectionPtr_t out;
        for (auto input : m_inputs)
        {
            out = input->registerUpdateNotifier(f);
            if (out)
            {
                break;
            }
        }
        return out;
    }

    ConnectionPtr_t IMultiInput::registerUpdateNotifier(const ISignalRelay::Ptr_t& relay)
    {
        // TODO
        std::vector<ConnectionPtr_t> connections;
        for (auto in : m_inputs)
        {
            auto connection = in->registerUpdateNotifier(relay);
            if (connection)
            {
                connections.push_back(connection);
            }
        }
        return std::make_shared<ConnectionSet>(std::move(connections));
    }

    bool IMultiInput::modified() const
    {
        if (m_current_input)
        {
            return m_current_input->modified();
        }
        for (auto in : m_inputs)
        {
            if (in->modified())
            {
                return true;
            }
        }
        return false;
    }

    void IMultiInput::modified(bool value)
    {
        for (auto in : m_inputs)
        {
            in->modified(value);
        }
    }

    MultiConnection::MultiConnection(std::vector<std::shared_ptr<Connection>>&& connections)
        : m_connections(connections)
    {
    }

    MultiConnection::~MultiConnection()
    {
    }

    bool MultiConnection::disconnect()
    {
        for (const auto& con : m_connections)
        {
            if (con)
            {
                con->disconnect();
            }
        }
        return true;
    }
} // namespace mo
