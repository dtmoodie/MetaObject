#pragma once
#include <MetaObject/thread/PriorityLevels.hpp>
#include <memory>

namespace mo
{
    class IAsyncStream;
    struct AsyncStreamConstructor
    {
        using Ptr_t = std::shared_ptr<IAsyncStream>;

        virtual ~AsyncStreamConstructor();

        virtual uint32_t priority(const int32_t device_id) = 0;
        virtual Ptr_t create(const std::string& name,
                             const int32_t device_id,
                             const PriorityLevels device_priority,
                             const PriorityLevels thread_priority) = 0;
    };
}
