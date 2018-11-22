#ifndef MO_CORE_ASYNC_STREAM_HPP
#define MO_CORE_ASYNC_STREAM_HPP
#include "MetaObject/core.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
#include <MetaObject/core/metaobject_config.hpp>
#include <MetaObject/thread/PriorityLevels.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

struct SystemTable;

namespace mo
{
    struct AsyncStreamConstructor;

    struct MO_EXPORTS AsyncStreamFactory
    {
        using Ptr_t = std::shared_ptr<IAsyncStream>;

        static AsyncStreamFactory* instance();
        static AsyncStreamFactory* instance(SystemTable* table);

        void registerConstructor(AsyncStreamConstructor* ctr);

        Ptr_t create(const std::string& name = "",
                     const int32_t device_id = 0,
                     const PriorityLevels device_priority = MEDIUM,
                     const PriorityLevels thread_priority = MEDIUM);

        void onStreamDestroy(const IAsyncStream* stream);

      private:
        std::vector<IAsyncStream*> m_streams;
        std::vector<AsyncStreamConstructor*> m_ctrs;
    }; // struct mo::AsyncStreamFactory

    class Allocator;
    class MO_EXPORTS IAsyncStream
    {
      public:
        using Ptr_t = std::shared_ptr<IAsyncStream>;

        virtual ~IAsyncStream();

        virtual void pushWork(std::function<void(void)>&& work) = 0;
        virtual void pushEvent(std::function<void(void)>&& event, const uint64_t event_id = 0) = 0;

        virtual void setName(const std::string& name) = 0;
        virtual void setHostPriority(const PriorityLevels p) = 0;
        virtual void setDevicePriority(const PriorityLevels p) = 0;

        virtual std::string name() const = 0;
        virtual uint64_t threadId() const = 0;
        virtual bool isDeviceContext() const = 0;
        virtual uint64_t processId() const = 0;
        virtual uint64_t streamId() const = 0;
        virtual AllocatorPtr_t hostAllocator() const = 0;
        virtual TypeInfo interface() const = 0;
    }; // class mo::IContext

    class MO_EXPORTS AsyncStream : virtual public IAsyncStream
    {
      public:
        using Ptr_t = std::shared_ptr<AsyncStream>;

        AsyncStream();
        ~AsyncStream() override;

        void pushWork(std::function<void(void)>&& work) override;
        void pushEvent(std::function<void(void)>&& event, const uint64_t event_id = 0) override;

        void setName(const std::string& name) override;
        void setHostPriority(const PriorityLevels p) override;
        void setDevicePriority(const PriorityLevels p) override;

        std::string name() const override;
        uint64_t threadId() const override;
        bool isDeviceContext() const override;
        uint64_t processId() const override;
        AllocatorPtr_t hostAllocator() const override;
        TypeInfo interface() const override;
        uint64_t streamId() const override;

      protected:
        AsyncStream(const TypeInfo derived_interface);

      private:
        std::string m_name;
        uint64_t m_process_id = 0;
        uint64_t m_thread_id = 0;
        int32_t m_device_id = -1;
        std::string m_host_name;
        AllocatorPtr_t m_allocator;
        TypeInfo m_type;
        uint64_t m_stream_id = 0;
        PriorityLevels m_host_priority = MEDIUM;
    }; // class mo::IContext

} // namespace mo

#endif // MO_CORE_ASYNC_STREAM_HPP
