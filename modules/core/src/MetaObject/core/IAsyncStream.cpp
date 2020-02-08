#include "IAsyncStream.hpp"
#include "AsyncStreamFactory.hpp"
#include <MetaObject/thread/FiberProperties.hpp>

#include <boost/fiber/fiber.hpp>
#include <boost/fiber/operations.hpp>
namespace mo
{
    IAsyncStream::~IAsyncStream() = default;

    IAsyncStream::Ptr_t IAsyncStream::create(const std::string& name,
                                             int32_t device_id,
                                             PriorityLevels device_priority,
                                             PriorityLevels thread_priority)
    {
        auto stream = AsyncStreamFactory::instance()->create(name, device_id, device_priority, thread_priority);
        setCurrent(stream);
        return stream;
    }

    IDeviceStream::Ptr_t IDeviceStream::create(const std::string& name,
                                               int32_t device_id,
                                               PriorityLevels device_priority,
                                               PriorityLevels thread_priority)
    {
        auto stream = IAsyncStream::create(name, device_id, device_priority, thread_priority);
        auto typed = std::dynamic_pointer_cast<IDeviceStream>(stream);
        setCurrent(typed);
        return typed;
    }

    auto IAsyncStream::current() -> Ptr_t
    {
        return boost::this_fiber::properties<FiberProperty>().stream();
    }

    void IAsyncStream::setCurrent(Ptr_t stream)
    {
        boost::this_fiber::properties<FiberProperty>().setStream(std::move(stream));
    }

    auto IDeviceStream::current() -> Ptr_t
    {
        return boost::this_fiber::properties<FiberProperty>().deviceStream();
    }

    void IDeviceStream::setCurrent(Ptr_t stream)
    {
        IAsyncStream::setCurrent(stream);
        boost::this_fiber::properties<FiberProperty>().setDeviceStream(std::move(stream));
    }
} // namespace mo
