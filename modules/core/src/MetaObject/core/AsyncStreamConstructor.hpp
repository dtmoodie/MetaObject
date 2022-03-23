#pragma once
#include <MetaObject/thread/PriorityLevels.hpp>
#include <memory>

namespace mo
{
    struct IAsyncStream;
    class WorkQueue;
    struct MO_EXPORTS AsyncStreamConstructor
    {
        using Ptr_t = std::shared_ptr<IAsyncStream>;

        AsyncStreamConstructor() = default;
        AsyncStreamConstructor(const AsyncStreamConstructor&) = default;
        AsyncStreamConstructor(AsyncStreamConstructor&&) = default;
        AsyncStreamConstructor& operator=(const AsyncStreamConstructor&) = default;
        AsyncStreamConstructor& operator=(AsyncStreamConstructor&&) = default;
        virtual ~AsyncStreamConstructor();

        virtual uint32_t priority(int32_t device_id) = 0;
        virtual Ptr_t create(const std::string& name,
                             int32_t device_id,
                             PriorityLevels device_priority,
                             PriorityLevels thread_priority) = 0;
    };
} // namespace mo
