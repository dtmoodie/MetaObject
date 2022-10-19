#ifndef MO_THREAD_FIBER_PROPERTIES_HPP
#define MO_THREAD_FIBER_PROPERTIES_HPP
#include "PriorityLevels.hpp"

#include <boost/fiber/properties.hpp>

#include <cstdint>
#include <memory>

namespace mo
{
    struct IAsyncStream;
    struct IDeviceStream;
    struct PriorityScheduler;
    struct MO_EXPORTS FiberProperty : public boost::fibers::fiber_properties
    {
        FiberProperty(boost::fibers::context* ctx);

        PriorityLevels getPriority() const;
        void setPriority(PriorityLevels p);

        uint64_t getId() const;
        void setId(uint64_t id);

        void setBoth(PriorityLevels priority, uint64_t id);
        void setAll(PriorityLevels priority, uint64_t id);
        void setAll(PriorityLevels priority, uint64_t id, IAsyncStreamPtr_t);

        void disable();
        bool isEnabled() const;

        IAsyncStreamPtr_t getStream() const;

        void setStream(IAsyncStreamPtr_t);

        void setScheduler(PriorityScheduler*);
        PriorityScheduler* getScheduler() const;

      private:
        PriorityLevels m_priority = MEDIUM;
        uint64_t m_id = 0;
        bool m_enabled = true;
        IAsyncStreamWeakPtr_t m_stream;
        PriorityScheduler* m_scheduler = nullptr;
    };
} // namespace mo

#endif // MO_THREAD_FIBER_PROPERTIES_HPP
