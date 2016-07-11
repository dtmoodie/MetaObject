#pragma once
#include <map>
#include <mutex>
#include <vector>
#include <signals/Defs.h>


namespace Signals
{
    size_t SIGNAL_EXPORTS get_this_thread();
    class SIGNAL_EXPORTS thread_registry
    {
        std::map<int, std::vector<size_t>> _thread_map;
        thread_registry();
        std::mutex mtx;
    public:
        virtual void register_thread(int type, size_t id = get_this_thread());

        virtual size_t get_thread(int type);
        static thread_registry* get_instance();
    };
}
