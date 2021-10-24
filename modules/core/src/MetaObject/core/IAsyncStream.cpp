#include "IAsyncStream.hpp"
#include "AsyncStreamFactory.hpp"
#include <MetaObject/thread/FiberProperties.hpp>
#include <MetaObject/thread/Thread.hpp>

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
        setCurrent(stream);
        auto typed = std::dynamic_pointer_cast<IDeviceStream>(stream);
        return typed;
    }

    IAsyncStream::Ptr_t IAsyncStream::current()
    {
        boost::fibers::context* ctx = boost::fibers::context::active();
        if (ctx)
        {
            return boost::this_fiber::properties<FiberProperty>().getStream();
        }
        return {};
    }

    IAsyncStream& IAsyncStream::currentRef()
    {
        auto cur = current();
        MO_ASSERT(cur != nullptr);
        return *cur;
    }

    void IAsyncStream::setCurrent(std::shared_ptr<IAsyncStream> stream)
    {
        boost::this_fiber::properties<FiberProperty>().setStream(stream);
    }

    IDeviceStream* IAsyncStream::getDeviceStream()
    {
        return nullptr;
    }

    const IDeviceStream* IAsyncStream::getDeviceStream() const
    {
        return nullptr;
    }

    IDeviceStream::Ptr_t IDeviceStream::current()
    {
        auto stream = boost::this_fiber::properties<FiberProperty>().getStream();
        return std::dynamic_pointer_cast<IDeviceStream>(stream);
    }

    IDeviceStream* IDeviceStream::getDeviceStream()
    {
        return this;
    }

    const IDeviceStream* IDeviceStream::getDeviceStream() const
    {
        return this;
    }
} // namespace mo
