#pragma once
#include <memory>

namespace mo
{
    struct Context;
    struct ContextConstructor
    {
        using Ptr = std::shared_ptr<Context>;

        virtual ~ContextConstructor();

        virtual int priority(int device_id, int device_priority, int thread_priority) = 0;
        virtual Ptr create(const std::string& name, int device_id, int device_priority, int thread_priority) = 0;
    };
}
