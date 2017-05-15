#pragma once
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/Thread/ThreadRegistry.hpp"

#include <functional>

namespace mo
{
    class MO_EXPORTS ThreadSpecificQueue
    {
    public:
        static void Push(const std::function<void(void)>& f, size_t id = GetThisThread(), void* obj = nullptr);
        static void RemoveFromQueue(void* obj);
        static int Run(size_t id = GetThisThread());
        static bool RunOnce(size_t id = GetThisThread());
        // Register a notifier function to signal new data input onto a queue
        static void RegisterNotifier(const std::function<void(void)>& f, size_t id = GetThisThread());
		static size_t Size(size_t id = GetThisThread());
        static void Cleanup();
    };
} // namespace Signals
