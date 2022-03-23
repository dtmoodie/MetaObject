#include "MetaObject/signals/TSignal.hpp"
#include "MetaObject/signals/detail/SignalMacros.hpp"
#include "MetaObject/signals/detail/SlotMacros.hpp"
#include "MetaObject/thread/ThreadRegistry.hpp"
#include <MetaObject/core/AsyncStreamFactory.hpp>

#include "RuntimeObjectSystem/IObjectFactorySystem.h"
#include "RuntimeObjectSystem/RuntimeObjectSystem.h"

#include <MetaObject/thread/Thread.hpp>
#include <MetaObject/thread/fiber_include.hpp>

#include <boost/thread.hpp>
#include <iostream>

#include "gtest/gtest.h"

using namespace mo;

TEST(signals, single_threaded)
{
    TSignal<int(int)> signal;
    {
        TSlot<int(int)> slot([](int val) { return val * 2; });
        signal.connect(slot);

        ASSERT_EQ(signal(4), 8);
        ASSERT_EQ(signal(8), 16);
    }
    ASSERT_THROW(signal(4), mo::TExceptionWithCallstack<std::runtime_error>);
}

TEST(signals, multi_threaded)
{
    auto stream = mo::AsyncStreamFactory::instance()->create();
    IAsyncStreamPtr_t thread_ctx;

    TSlot<void(int)> slot = TSlot<void(int)>(std::bind(
        [&thread_ctx](int value) -> void {
            ASSERT_EQ(thread_ctx->threadId(), mo::getThisThread());
            ASSERT_EQ(5, value);
        },
        std::placeholders::_1));

    boost::thread thread = boost::thread([&thread_ctx]() -> void {
        mo::initThread();
        thread_ctx = mo::AsyncStreamFactory::instance()->create("Thread context");
        while (!boost::this_thread::interruption_requested())
        {
            // TODO
            // ThreadSpecificQueue::run(thread_ctx->threadId());
        }
    });
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));

    slot.setStream(*thread_ctx);

    TSignal<void(int)> signal;
    auto Connection = slot.connect(signal);

    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
    signal(stream.get(), 5);
    thread.interrupt();
    thread.join();
}
