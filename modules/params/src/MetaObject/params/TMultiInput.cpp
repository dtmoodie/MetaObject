#include "TMultiInput-inl.hpp"
#include <boost/thread/recursive_mutex.hpp>

namespace mo
{

    IMultiInput::IMultiInput(const std::vector<InputParam*>& inputs)
        : m_inputs(inputs)
        , IParam("", mo::ParamFlags::Input_e)
    {
    }

    bool IMultiInput::setInput(const std::shared_ptr<IParam>& publisher)
    {
        Lock lock(mtx());
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
        Lock lock(mtx());
        for (auto input : m_inputs)
        {
            if (input->acceptsInput(publisher))
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

    bool IMultiInput::getInputData(const Header& desired, Header* retrieved)
    {
        Lock lock(mtx());
        if (m_current_input)
        {
            return m_current_input->getInputData(desired, retrieved);
        }
        return false;
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
        Lock lock(mtx());
        if (m_current_input)
        {
            return m_current_input->getTypeInfo();
        }
        else
        {
            return _void_type_info;
        }
    }

    const mo::TypeInfo IMultiInput::_void_type_info = mo::TypeInfo(typeid(void));

    mo::IParam* IMultiInput::getInputParam() const
    {
        return m_current_input;
    }

    OptionalTime_t IMultiInput::getInputTimestamp()
    {
        if (m_current_input)
        {
            return m_current_input->getInputTimestamp();
        }
        return {};
    }

    uint64_t IMultiInput::getInputFrameNumber()
    {
        if (m_current_input)
        {
            return m_current_input->getInputFrameNumber();
        }
        return std::numeric_limits<uint64_t>::max();
    }

    OptionalTime_t IMultiInput::getTimestamp() const
    {
    }
    uint64_t IMultiInput::getFrameNumber() const
    {
    }

    bool IMultiInput::isInputSet() const
    {
    }

    bool IMultiInput::acceptsInput(IParam* input) const
    {
    }

    bool IMultiInput::acceptsType(const TypeInfo& type) const
    {
    }

    ConnectionPtr_t IMultiInput::registerUpdateNotifier(ISlot* f)
    {
    }

    ConnectionPtr_t IMultiInput::registerUpdateNotifier(const ISignalRelay::Ptr& relay)
    {
    }

    bool IMultiInput::modified() const
    {
    }

    void modified(bool value)
    {
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
}
