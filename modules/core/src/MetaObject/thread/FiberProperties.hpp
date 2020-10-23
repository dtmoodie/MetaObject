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
    struct MO_EXPORTS FiberProperty : public boost::fibers::fiber_properties
    {
        FiberProperty(boost::fibers::context* ctx);

        PriorityLevels getPriority() const;
        void setPriority(PriorityLevels p);

        uint64_t getId() const;
        void setId(uint64_t id);

        void setBoth(PriorityLevels priority, uint64_t id);
        void setAll(PriorityLevels priority, uint64_t id, bool work);

        bool isWork() const;
        void setWork(bool val);

        void disable();
        bool enabled() const;

        std::shared_ptr<IAsyncStream> stream() const;
        std::shared_ptr<IAsyncStream> stream();
        std::shared_ptr<IDeviceStream> deviceStream() const;

        void setStream(std::shared_ptr<IAsyncStream>);
        void setDeviceStream(std::shared_ptr<IDeviceStream>);

      private:
        PriorityLevels m_priority = MEDIUM;
        uint64_t m_id = 0;
        bool m_enabled = true;
        bool m_is_work = true;
        // Weak pointer or shared ptr?
        std::weak_ptr<IAsyncStream> m_stream;
        std::weak_ptr<IDeviceStream> m_device_stream;
    };
} // namespace mo

#endif // MO_THREAD_FIBER_PROPERTIES_HPP
