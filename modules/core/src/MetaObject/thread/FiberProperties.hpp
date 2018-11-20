#ifndef MO_THREAD_FIBER_PROPERTIES_HPP
#define MO_THREAD_FIBER_PROPERTIES_HPP
#include "PriorityLevels.hpp"

#include <boost/fiber/properties.hpp>
#include <cstdint>

namespace mo
{
    struct FiberProperty : public boost::fibers::fiber_properties
    {
        FiberProperty(boost::fibers::context* ctx);

        PriorityLevels getPriority() const;
        void setPriority(const PriorityLevels p);

        uint64_t getId() const;
        void setId(const uint64_t id);

        void setBoth(const PriorityLevels priority, const uint64_t id);

      private:
        PriorityLevels m_priority = MEDIUM;
        uint64_t m_id = 0;
    };
}

#endif // MO_THREAD_FIBER_PROPERTIES_HPP
