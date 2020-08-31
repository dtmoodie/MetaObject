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
        auto typed = std::dynamic_pointer_cast<IDeviceStream>(stream);
        setCurrent(typed);
        return typed;
    }

    auto IAsyncStream::current() -> Ptr_t
    {
        boost::fibers::context* ctx = boost::fibers::context::active();
        if (ctx)
        {
            auto props = ctx->get_properties();
            if (!props)
            {
                initThread();
            }
            return boost::this_fiber::properties<FiberProperty>().stream();
        }
        return {};
    }

    IAsyncStream& IAsyncStream::currentRef()
    {
        auto cur = current();
        MO_ASSERT(cur != nullptr);
        return *cur;
    }

    void IAsyncStream::setCurrent(Ptr_t stream)
    {
        auto props = boost::fibers::context::active()->get_properties();
        if (!props)
        {
            initThread();
        }

        boost::this_fiber::properties<FiberProperty>().setStream(std::move(stream));
    }

    IDeviceStream* IAsyncStream::getDeviceStream()
    {
        return nullptr;
    }

    const IDeviceStream* IAsyncStream::getDeviceStream() const
    {
        return nullptr;
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

    IDeviceStream* IDeviceStream::getDeviceStream()
    {
        return this;
    }

    const IDeviceStream* IDeviceStream::getDeviceStream() const
    {
        return this;
    }
} // namespace mo
