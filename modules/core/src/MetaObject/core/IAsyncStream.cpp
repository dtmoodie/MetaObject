#include "IAsyncStream.hpp"
#include "AsyncStreamFactory.hpp"
#include <MetaObject/thread/FiberProperties.hpp>
#include <MetaObject/thread/Thread.hpp>

#include <boost/fiber/fiber.hpp>
#include <boost/fiber/operations.hpp>
namespace mo
{

    AsyncStreamContextManager::AsyncStreamContextManager(const std::shared_ptr<IAsyncStream>& new_stream)
    {
        m_previous = IAsyncStream::current();
        IAsyncStream::setCurrent(new_stream);
    }

    AsyncStreamContextManager::~AsyncStreamContextManager()
    {
        IAsyncStream::setCurrent(m_previous);
    }

    IAsyncStream::~IAsyncStream() = default;

    IAsyncStream::Ptr_t IAsyncStream::create(const std::string& name,
                                             int32_t device_id,
                                             PriorityLevels device_priority,
                                             PriorityLevels thread_priority)
    {
        std::shared_ptr<AsyncStreamFactory> instance = AsyncStreamFactory::instance();
        auto stream = instance->create(name, device_id, device_priority, thread_priority);
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

    void IAsyncStream::makeCurrent()
    {

    }

    void IAsyncStream::noLongerCurrent()
    {

    }

    void IAsyncStream::waitForCompletion()
    {
        this->synchronize();
    }

    // TODO implement
    void IAsyncStream::synchronize(IAsyncStream& other)
    {
        std::shared_ptr<ConditionVariable> cv = std::make_shared<ConditionVariable>();
        // This is needed in case the other stream has already finished all of its work before we try to have this stream wait on results
        // Otherwise we will wait indefinitely on a condition variable that has already notified
        std::shared_ptr<bool> triggered = std::make_shared<bool>(false);
        other.pushWork([cv, triggered](IAsyncStream* stream)
        {
            cv->notify_all();
            *triggered = true;
        });
        this->pushWork([cv, triggered](IAsyncStream* stream)
        {
            if(!*triggered)
            {
                Mutex mtx;
                Mutex::Lock_t lock(mtx);
                cv->wait(lock);
            }

        });
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
