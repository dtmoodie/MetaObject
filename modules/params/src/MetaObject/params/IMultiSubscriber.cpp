#include "IMultiSubscriber.hpp"

#include <MetaObject/signals/Connection.hpp>
#include <MetaObject/thread/Mutex.hpp>
#include <MetaObject/thread/fiber_include.hpp>

namespace mo
{

    IMultiSubscriber::IMultiSubscriber()
    {
    }

    void IMultiSubscriber::setInputs(const std::vector<ISubscriber*>& inputs)
    {
        m_inputs = inputs;
    }

    bool IMultiSubscriber::setInput(std::shared_ptr<IPublisher> publisher)
    {
        MO_ASSERT(publisher != nullptr);
        Lock_t lock(mtx());
        for (auto input : m_inputs)
        {
            if (input->acceptsPublisher(*publisher))
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

    bool IMultiSubscriber::setInput(IPublisher* publisher)
    {
        Lock_t lock(mtx());
        for (auto input : m_inputs)
        {
            if (publisher && input->acceptsPublisher(*publisher))
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

    /*IDataContainerConstPtr_t IMultiSubscriber::getInputData(IAsyncStream* stream) const
    {
        Lock_t lock(mtx());
        auto input = m_current_input;
        lock.unlock();
        if (input)
        {
            return input->getInputData(stream);
        }
        return {};
    }*/

    IDataContainerConstPtr_t IMultiSubscriber::getCurrentData(IAsyncStream* stream) const
    {
        Lock_t lock(mtx());
        auto input = m_current_input;
        lock.unlock();
        if (input)
        {
            return input->getCurrentData(stream);
        }
        return {};
    }

    IDataContainerConstPtr_t IMultiSubscriber::getData(const Header* desired, IAsyncStream* stream)
    {
        Lock_t lock(mtx());
        auto input = m_current_input;
        lock.unlock();
        if (input)
        {
            return input->getData(desired, stream);
        }
        return {};
    }

    void IMultiSubscriber::setMtx(Mutex_t& mtx)
    {
        for (auto input : m_inputs)
        {
            input->setMtx(mtx);
        }
    }

    IPublisher* IMultiSubscriber::getPublisher() const
    {
        if (m_current_input)
        {
            return m_current_input->getPublisher();
        }
        return nullptr;
    }

    bool IMultiSubscriber::isInputSet() const
    {
        return m_current_input != nullptr;
    }

    bool IMultiSubscriber::acceptsPublisher(const IPublisher& input) const
    {
        for (auto in : m_inputs)
        {
            if (in->acceptsPublisher(input))
            {
                return true;
            }
        }
        return false;
    }

    bool IMultiSubscriber::acceptsType(const TypeInfo& type) const
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

    ConnectionPtr_t IMultiSubscriber::registerUpdateNotifier(ISlot& f)
    {
        std::vector<ConnectionPtr_t> connections;
        for (auto input : m_inputs)
        {
            auto connection = input->registerUpdateNotifier(f);
            if (connection)
            {
                connections.push_back(std::move(connection));
            }
        }
        return std::make_shared<ConnectionSet>(std::move(connections));
    }

    ConnectionPtr_t IMultiSubscriber::registerUpdateNotifier(const ISignalRelay::Ptr_t& relay)
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

    std::ostream& IMultiSubscriber::print(std::ostream& os) const
    {
        os << this->getTreeName();
        os << '\n';
        for (auto in : m_inputs)
        {
            os << ' ';
            in->print(os);
            os << '\n';
        }

        return os;
    }

    bool IMultiSubscriber::hasNewData() const
    {
        for (auto in : m_inputs)
        {
            if (in->hasNewData())
            {
                return true;
            }
        }
        return false;
    }

    std::vector<Header> IMultiSubscriber::getAvailableHeaders() const
    {
        std::vector<Header> output;
        for (auto in : m_inputs)
        {
            auto tmp = in->getAvailableHeaders();
            output.insert(tmp.begin(), tmp.end(), output.end());
        }
        return output;
    }

    boost::optional<Header> IMultiSubscriber::getNewestHeader() const
    {
        boost::optional<Header> out;
        for (auto in : m_inputs)
        {
            const auto tmp = in->getNewestHeader();
            if (tmp && !out)
            {
                out = tmp;
            }
            if (tmp && out)
            {
                if (tmp->timestamp && !out->timestamp)
                {
                    out = tmp;
                }
                else
                {
                    if (tmp->timestamp && out->timestamp)
                    {
                        if (*tmp->timestamp > *out->timestamp)
                        {
                            out = tmp;
                        }
                    }
                    else
                    {
                        if (tmp->frame_number > out->frame_number)
                        {
                            out = tmp;
                        }
                    }
                }
            }
        }
        return out;
    }

    TypeInfo IMultiSubscriber::getCurrentInputType() const
    {
        if (m_current_input)
        {
            auto outs = m_current_input->getInputTypes();
            if (!outs.empty())
            {
                return outs[0];
            }
        }
        return TypeInfo::Void();
    }

} // namespace mo
