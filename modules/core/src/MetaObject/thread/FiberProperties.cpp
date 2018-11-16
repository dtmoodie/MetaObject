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
}
