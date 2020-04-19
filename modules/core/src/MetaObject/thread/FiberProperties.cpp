#include "FiberProperties.hpp"

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

    void FiberProperty::setAll(const PriorityLevels priority, const uint64_t id, const bool work)
    {
        if (m_priority != priority || id != m_id || m_is_work != work)
        {
            m_priority = priority;
            m_id = id;
            m_is_work = work;
            notify();
        }
    }

    bool FiberProperty::isWork() const
    {
        return m_is_work;
    }

    void FiberProperty::setWork(const bool val)
    {
        if (m_is_work != val)
        {
            m_is_work = val;
            notify();
        }
    }

    void FiberProperty::disable()
    {
        m_enabled = false;
    }

    bool FiberProperty::enabled() const
    {
        return m_enabled;
    }

    std::shared_ptr<IAsyncStream> FiberProperty::stream() const
    {
        return m_stream.lock();
    }

    std::shared_ptr<IDeviceStream> FiberProperty::deviceStream() const
    {
        return m_device_stream.lock();
    }

    void FiberProperty::setStream(std::shared_ptr<IAsyncStream> stream)
    {
        m_stream = std::move(stream);
    }

    void FiberProperty::setDeviceStream(std::shared_ptr<IDeviceStream> stream)
    {
        m_device_stream = std::move(stream);
    }
} // namespace mo
