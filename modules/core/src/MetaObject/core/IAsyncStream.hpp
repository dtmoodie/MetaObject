#ifndef MO_CORE_IASYNC_STREAM_HPP
#define MO_CORE_IASYNC_STREAM_HPP
#include <MetaObject/thread/PriorityLevels.hpp>
#include <ct/types/TArrayView.hpp>
#include <functional>

namespace mo
{
    struct Allocator;
    struct IDeviceStream;
    struct IAsyncStream;

    struct MO_EXPORTS IAsyncStream
    {
        using Ptr_t = std::shared_ptr<IAsyncStream>;

        static Ptr_t current();
        static IAsyncStream& currentRef();
        static void setCurrent(Ptr_t);

        static Ptr_t create(const std::string& name = "",
                            int32_t device_id = 0,
                            PriorityLevels device_priority = MEDIUM,
                            PriorityLevels thread_priority = MEDIUM);

        IAsyncStream() = default;
        IAsyncStream(const IAsyncStream&) = delete;
        IAsyncStream(IAsyncStream&&) = delete;
        IAsyncStream& operator=(const IAsyncStream&) = delete;
        IAsyncStream& operator=(IAsyncStream&&) = delete;

        virtual ~IAsyncStream();

        virtual void pushWork(std::function<void(void)>&& work, PriorityLevels priority = NONE) = 0;
        virtual void
        pushEvent(std::function<void(void)>&& event, uint64_t event_id = 0, PriorityLevels priority = NONE) = 0;

        virtual void synchronize() = 0;

        virtual void setName(const std::string& name) = 0;
        virtual void setHostPriority(PriorityLevels p) = 0;

        virtual std::string name() const = 0;
        virtual uint64_t threadId() const = 0;
        virtual bool isDeviceStream() const = 0;
        virtual IDeviceStream* getDeviceStream();
        virtual const IDeviceStream* getDeviceStream() const;
        virtual uint64_t processId() const = 0;
        virtual uint64_t streamId() const = 0;
        virtual AllocatorPtr_t hostAllocator() const = 0;
    }; // class mo::IContext

    struct MO_EXPORTS IDeviceStream : virtual public IAsyncStream
    {
        using Ptr_t = std::shared_ptr<IDeviceStream>;
        static Ptr_t current();
        static void setCurrent(Ptr_t);
        static Ptr_t create(const std::string& name = "",
                            int32_t device_id = 0,
                            PriorityLevels device_priority = MEDIUM,
                            PriorityLevels thread_priority = MEDIUM);

        virtual void hostToDevice(ct::TArrayView<void> dst, ct::TArrayView<const void> src) = 0;
        virtual void deviceToHost(ct::TArrayView<void> dst, ct::TArrayView<const void> src) = 0;
        virtual void deviceToDevice(ct::TArrayView<void> dst, ct::TArrayView<const void> src) = 0;
        virtual void hostToHost(ct::TArrayView<void> dst, ct::TArrayView<const void> src) = 0;

        virtual void synchronize() override = 0;
        virtual void synchronize(IDeviceStream* other) = 0;

        virtual IDeviceStream* getDeviceStream() override;
        virtual const IDeviceStream* getDeviceStream() const override;

        virtual std::shared_ptr<DeviceAllocator> deviceAllocator() const = 0;
    };
} // namespace mo
#endif // MO_CORE_IASYNC_STREAM_HPP
