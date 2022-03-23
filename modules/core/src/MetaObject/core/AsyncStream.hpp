#ifndef MO_CORE_ASYNC_STREAM_HPP
#define MO_CORE_ASYNC_STREAM_HPP
#include "IAsyncStream.hpp"
#include <MetaObject/core.hpp>
#include <MetaObject/core/detail/Allocator.hpp>
#include <MetaObject/core/metaobject_config.hpp>
#include <MetaObject/detail/TypeInfo.hpp>

#include <boost/fiber/fiber.hpp>
#include <boost/fiber/mutex.hpp>

#include <memory>
#include <string>
#include <vector>

struct SystemTable;

namespace mo
{
    struct AsyncStreamConstructor;

    struct MO_EXPORTS AsyncStream : virtual public IAsyncStream, public std::enable_shared_from_this<AsyncStream>
    {
        using Ptr_t = std::shared_ptr<AsyncStream>;

        AsyncStream(AllocatorPtr_t host_alloc = Allocator::getDefault());
        AsyncStream(const AsyncStream&) = delete;
        AsyncStream(AsyncStream&&) = delete;
        AsyncStream& operator=(const AsyncStream&) = delete;
        AsyncStream& operator=(AsyncStream&&) = delete;
        ~AsyncStream() override;

        void pushWork(std::function<void(IAsyncStream*)>&& work) override;
        void pushEvent(std::function<void(IAsyncStream*)>&& event, uint64_t event_id = 0) override;

        void setName(const std::string& name) override;
        void setHostPriority(PriorityLevels p) override;
        PriorityLevels hostPriority() const;
        void synchronize() override;
        void hostToHost(ct::TArrayView<void> dst, ct::TArrayView<const void> src) override;

        std::string name() const override;
        uint64_t threadId() const override;
        bool isDeviceStream() const override;
        uint64_t processId() const override;
        AllocatorPtr_t hostAllocator() const override;
        uint64_t streamId() const override;
        size_t size() const override;

        void initializeFiber();

      private:
        std::string m_name;
        uint64_t m_process_id = 0;
        uint64_t m_thread_id = 0;
        int32_t m_device_id = -1;
        std::string m_host_name;
        AllocatorPtr_t m_allocator;
        uint64_t m_stream_id = 0;
        PriorityLevels m_host_priority = MEDIUM;
        std::deque<std::tuple<std::function<void(IAsyncStream*)>, uint64_t>> m_work_queue;
        boost::fibers::fiber m_worker_fiber;
        std::atomic<bool> m_continue;
        mutable boost::fibers::mutex m_mtx;

        static void workerLoop(AsyncStream*);
    }; // class mo::IContext

} // namespace mo

#endif // MO_CORE_ASYNC_STREAM_HPP
