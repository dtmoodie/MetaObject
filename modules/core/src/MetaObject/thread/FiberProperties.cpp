#include "FiberProperties.hpp"
#include <MetaObject/core/AsyncStreamFactory.hpp>
#include <MetaObject/core/IAsyncStream.hpp>
namespace mo
{

    FiberProperty::FiberProperty(boost::fibers::context* ctx)
        : boost::fibers::fiber_properties(ctx)
    {
    }

    PriorityLevels FiberProperty::getPriority() const
    {
        return m_priority;
    }

    void FiberProperty::setPriority(const PriorityLevels p)
    {
        if (p != m_priority)
        {
            m_priority = p;
            notify();
        }
    }

    uint64_t FiberProperty::getId() const
    {
        return m_id;
    }

    void FiberProperty::setId(const uint64_t id)
    {
        if (id != m_id)
        {
            m_id = id;
            notify();
        }
    }

    void FiberProperty::setBoth(const PriorityLevels priority, const uint64_t id)
    {
        if (m_priority != priority || id != m_id)
        {
            m_priority = priority;
            m_id = id;
            notify();
        }
    }

    void FiberProperty::setAll(const PriorityLevels priority, const uint64_t id)
    {
        if (m_priority != priority || id != m_id)
        {
            m_priority = priority;
            m_id = id;
            notify();
        }
    }
    void FiberProperty::setAll(PriorityLevels priority, uint64_t id, std::shared_ptr<IAsyncStream> stream)
    {
        if (m_priority != priority || id != m_id)
        {
            m_priority = priority;
            m_id = id;
            m_stream = std::move(stream);
            notify();
        }
    }

    void FiberProperty::disable()
    {
        // Instead of calling ready_unlink on the context, we set a flag to disable execution of the lambda
        // this->ctx_->ready_unlink();
        m_enabled = false;
    }

    bool FiberProperty::isEnabled() const
    {
        return m_enabled;
    }

    std::shared_ptr<IAsyncStream> FiberProperty::getStream() const
    {
        return m_stream.lock();
    }

    void FiberProperty::setStream(std::shared_ptr<IAsyncStream> stream)
    {
        m_stream = std::move(stream);
        notify();
    }
} // namespace mo
