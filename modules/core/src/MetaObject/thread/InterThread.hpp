#pragma once
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/thread/ThreadRegistry.hpp"

#include <functional>

namespace mo
{
    class MO_EXPORTS ThreadSpecificQueue
    {
    public:
        struct MO_EXPORTS ScopedNotifier{
            ScopedNotifier(size_t tid);
            ~ScopedNotifier();
        private:
            size_t m_tid;
        };

        static void push(const std::function<void(void)>& f, size_t id = getThisThread(), void* obj = nullptr);
        static void removeFromQueue(void* obj);
        static int  run(size_t id = getThisThread());
        static bool runOnce(size_t id = getThisThread());
        // Register a notifier function to signal new data input onto a queue
        static ScopedNotifier registerNotifier(const std::function<void(void)>& f, size_t id = getThisThread());
        static void deregisterNotifier(size_t id = getThisThread());
        static size_t size(size_t id = getThisThread());
        static void cleanup(size_t id = getThisThread());
    };
} // namespace Signals
