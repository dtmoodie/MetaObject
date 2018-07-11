#include "MetaObject/signals/TSlot.hpp"
#include "MetaObject/thread/ThreadHandle.hpp"
#include "MetaObject/thread/ThreadPool.hpp"

int main()
{
    int call_count = 0;
    mo::TSlot<int(void)> inner_loop(std::bind([&call_count]() -> int {
        ++call_count;
        return 100;
    }));
    {
        mo::ThreadHandle handle = mo::ThreadPool::instance()->requestThread();
        mo::ThreadHandle handle2 = handle;
        auto connection = handle.setInnerLoop(&inner_loop);
        handle.start();
        boost::this_thread::sleep_for(boost::chrono::seconds(10));
        handle.stop();
    }
    boost::this_thread::sleep_for(boost::chrono::seconds(10));
    mo::ThreadPool::instance()->cleanup();
    return 0;
}