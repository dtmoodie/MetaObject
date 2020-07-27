#include "MetaObject/params/SubscriberAny.hpp"
#include "IDataContainer.hpp"
#include <MetaObject/thread/fiber_include.hpp>
#include <boost/thread/locks.hpp>

namespace mo
{

    SubscriberAny::SubscriberAny(const std::string& name)
    {
        this->setName(name);
        this->appendFlags(ParamFlags::kINPUT);
    }

    bool SubscriberAny::setInput(std::shared_ptr<IPublisher> param)
    {
        m_shared_publisher = param;
        m_publisher = param.get();
        return true;
    }

    bool SubscriberAny::setInput(IPublisher* param)
    {
        m_shared_publisher.reset();
        m_publisher = param;
        return true;
    }

    bool SubscriberAny::acceptsPublisher(const IPublisher&) const
    {
        return true;
    }

    bool SubscriberAny::acceptsType(const TypeInfo&) const
    {
        return true;
    }

    std::vector<TypeInfo> SubscriberAny::getInputTypes() const
    {
        return {};
    }

    bool SubscriberAny::isInputSet() const
    {
        return m_publisher != nullptr;
    }

    IDataContainerConstPtr_t SubscriberAny::getCurrentData(IAsyncStream* stream) const
    {
        IDataContainerConstPtr_t output;
        if (m_publisher)
        {
            output = m_publisher->getData(nullptr, stream);
        }
        return output;
    }

    bool SubscriberAny::hasNewData() const
    {
        if (!m_publisher)
        {
            return false;
        }
        return true;
    }

    boost::optional<Header> SubscriberAny::getNewestHeader() const
    {
        boost::optional<Header> output;
        if (m_publisher)
        {
            output = m_publisher->getNewestHeader();
        }
        return output;
    }

    IDataContainerConstPtr_t SubscriberAny::getData(const Header* desired, IAsyncStream* stream)
    {
        IDataContainerConstPtr_t output;
        if (m_publisher)
        {
            output = m_publisher->getData(desired, stream);
        }
        return output;
    }

    std::vector<Header> SubscriberAny::getAvailableHeaders() const
    {
        std::vector<Header> output;
        if (m_publisher)
        {
            output = m_publisher->getAvailableHeaders();
        }
        return output;
    }

    std::ostream& SubscriberAny::print(std::ostream& os) const
    {
        if (m_publisher)
        {
            m_publisher->print(os);
        }
        return os;
    }

    IPublisher* SubscriberAny::getPublisher() const
    {
        return m_publisher;
    }

} // namespace mo