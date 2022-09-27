#include "MetaObject/signals/TSignal.hpp"
#include "MetaObject/signals/detail/SignalMacros.hpp"
#include "MetaObject/signals/detail/SlotMacros.hpp"
#include "MetaObject/thread/ThreadRegistry.hpp"
#include <MetaObject/core/AsyncStreamFactory.hpp>

#include "RuntimeObjectSystem/IObjectFactorySystem.h"
#include "RuntimeObjectSystem/RuntimeObjectSystem.h"

#include <MetaObject/thread/FiberScheduler.hpp>
#include <MetaObject/thread/Thread.hpp>
#include <MetaObject/thread/ThreadPool.hpp>
#include <MetaObject/thread/fiber_include.hpp>

#include <boost/thread.hpp>
#include <iostream>

#include "gtest/gtest.h"

using namespace mo;

TEST(signal, single_threaded)
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

TEST(signal, multi_threaded)
{
    auto stream = mo::AsyncStream::create();
    mo::Thread thread;
    IAsyncStreamPtr_t thread_stream = thread.asyncStream();

    bool slot_invoked = false;
    TSlot<void(int)> slot = TSlot<void(int)>(std::bind(
        [&thread_stream, &slot_invoked](int value) -> void {
            ASSERT_EQ(thread_stream->threadId(), mo::getThisThread());
            ASSERT_EQ(5, value);
            slot_invoked = true;
        },
        std::placeholders::_1));

    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));

    slot.setStream(thread_stream.get());

    TSignal<void(int)> signal;
    auto connection = slot.connect(signal);

    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
    signal(stream.get(), 5);
    thread_stream->waitForCompletion();
    ASSERT_TRUE(slot_invoked);
}
