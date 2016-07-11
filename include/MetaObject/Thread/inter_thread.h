#pragma once
#include "Defs.h"
#include "thread_registry.h"

#include <functional>

namespace Signals
{
    class SIGNAL_EXPORTS thread_specific_queue
    {
    public:
        static void push(const std::function<void(void)>& f, size_t id = get_this_thread(), void* obj = nullptr);
        static void remove_from_queue(void* obj);
        static void run(size_t id = get_this_thread());
        static void run_once(size_t id = get_this_thread());
        // Register a notifier function to signal new data input onto a queue
        static void register_notifier(const std::function<void(void)>& f, size_t id = get_this_thread());
    };
} // namespace Signals
