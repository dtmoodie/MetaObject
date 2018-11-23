#include "ContextQueue.hpp"

namespace mo
{
    uint64_t ContextQueue::push(boost::fibers::context* c, const PriorityLevels priority)
    {
        m_queues[priority].push(c);
        uint64_t sz = 0;
        for (uint32_t i = 0; i <= HIGHEST; ++i)
        {
            sz += m_queues[i].size();
        }
        return sz;
    }

    boost::fibers::context* ContextQueue::pop()
    {
        boost::fibers::context* output = nullptr;
        for (int32_t i = HIGHEST; i >= 0; --i)
        {
            output = m_queues[i].pop();
            if (output)
            {
                break;
            }
        }
        return output;
    }

    boost::fibers::context* ContextQueue::steal()
    {
        boost::fibers::context* output = nullptr;
        for (int32_t i = HIGHEST; i >= 0; --i)
        {
            output = m_queues[i].steal();
            if (output)
            {
                break;
            }
        }
        return output;
    }

    bool ContextQueue::empty() const
    {
        for (uint32_t i = 0; i <= HIGHEST; ++i)
        {
            if (!m_queues[i].empty())
            {
                return false;
            }
        }
        return true;
    }
}
