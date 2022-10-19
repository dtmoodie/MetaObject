#include "IAsyncStream.hpp"
#include "AsyncStreamFactory.hpp"
#include <MetaObject/thread/FiberProperties.hpp>
#include <MetaObject/thread/Thread.hpp>

#include <boost/fiber/barrier.hpp>
#include <boost/fiber/fiber.hpp>
#include <boost/fiber/operations.hpp>

namespace mo
{

    AsyncStreamContextManager::AsyncStreamContextManager(const IAsyncStreamPtr_t& new_stream)
    {
        m_previous = IAsyncStream::current();
        IAsyncStream::setCurrent(new_stream);
    }

    AsyncStreamContextManager::~AsyncStreamContextManager()
    {
        IAsyncStream::setCurrent(m_previous);
    }

    IAsyncStream::~IAsyncStream()
    {
        if (IAsyncStream::current().get() == this)
        {
            IAsyncStream::setCurrent({});
        }
    }

    IAsyncStreamPtr_t IAsyncStream::create(const std::string& name,
                                           int32_t device_id,
                                           PriorityLevels device_priority,
                                           PriorityLevels thread_priority)
    {
        std::shared_ptr<AsyncStreamFactory> instance = AsyncStreamFactory::instance();
        auto stream = instance->create(name, device_id, device_priority, thread_priority);
        return stream;
        // IAsyncStreamPtr_t output(stream.get(), [stream](IAsyncStream* ptr) { stream->waitForCompletion(); });
        // return output;
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

    IAsyncStreamPtr_t IAsyncStream::current()
    {
        boost::fibers::context* ctx = boost::fibers::context::active();
        if (ctx)
        {
            FiberProperty& properties = boost::this_fiber::properties<FiberProperty>();
            auto stream = properties.getStream();
            return stream;
        }
        return {};
    }

    IAsyncStream& IAsyncStream::currentRef()
    {
        auto cur = current();
        MO_ASSERT(cur != nullptr);
        return *cur;
    }

    void IAsyncStream::setCurrent(IAsyncStreamPtr_t stream)
    {
        if (stream != nullptr)
        {
            // std::cout << "Setting stream '" << stream->name() << "' to current for fiber 0x"
            //          << boost::this_fiber::get_id() << " on thread " << boost::this_thread::get_id() << std::endl;
        }
        else
        {
            std::cout << "Setting stream 'null' to current for fiber 0x" << boost::this_fiber::get_id() << " on thread "
                      << boost::this_thread::get_id() << std::endl;
        }
        boost::fibers::context* ctx = boost::fibers::context::active();
        (void)ctx;
        FiberProperty& properties = boost::this_fiber::properties<FiberProperty>();
        properties.setStream(std::move(stream));
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

    void IAsyncStream::synchronize()
    {
        this->synchronize(1 * ns);
    }

    void IAsyncStream::synchronize(IAsyncStream& other)
    {
        const size_t other_size = other.size();
        const size_t my_size = this->size();
        if (other_size > 0)
        {
            std::shared_ptr<boost::fibers::barrier> barrier = std::make_shared<boost::fibers::barrier>(2);
            other.pushWork([barrier](IAsyncStream* stream) { barrier->wait(); });
            this->pushWork([barrier](IAsyncStream* stream) { barrier->wait(); });
        }
    }

    IDeviceStream* IAsyncStream::getDeviceStream()
    {
        return nullptr;
    }

    const IDeviceStream* IAsyncStream::getDeviceStream() const
    {
        return nullptr;
    }

    void IDeviceStream::synchronize()
    {
        this->synchronize(1 * ns);
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
