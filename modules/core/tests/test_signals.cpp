#include "MetaObject/core/detail/Counter.hpp"

#include "MetaObject/signals/TSignal.hpp"
#include "MetaObject/signals/detail/SignalMacros.hpp"
#include "MetaObject/signals/detail/SlotMacros.hpp"
#include "MetaObject/thread/ThreadRegistry.hpp"

#include "RuntimeObjectSystem/IObjectFactorySystem.h"
#include "RuntimeObjectSystem/RuntimeObjectSystem.h"

#include <boost/fiber/mutex.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

#include <boost/thread.hpp>
#include <iostream>

using namespace mo;

BOOST_AUTO_TEST_CASE(signals)
{
    TSignal<int(int)> signal;
    {
        TSlot<int(int)> slot([](int val) { return val * 2; });
        signal.connect(&slot);

        BOOST_CHECK_EQUAL(signal(4), 8);
        BOOST_CHECK_EQUAL(signal(8), 16);
    }
    BOOST_CHECK_THROW(signal(4), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(threaded_signal)
{
    auto stream = mo::AsyncStreamFactory::instance()->create();
    IAsyncStreamPtr_t thread_ctx;

    TSlot<void(int)> slot = TSlot<void(int)>(std::bind(
        [&thread_ctx](int value) -> void {
            BOOST_REQUIRE_EQUAL(thread_ctx->threadId(), mo::getThisThread());
            BOOST_REQUIRE_EQUAL(5, value);
        },
        std::placeholders::_1));

    boost::thread thread = boost::thread([&thread_ctx]() -> void {
        thread_ctx = mo::AsyncStreamFactory::instance()->create("Thread context");
        while (!boost::this_thread::interruption_requested())
        {
            // TODO
            // ThreadSpecificQueue::run(thread_ctx->threadId());
        }
    });
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));

    slot.setStream(thread_ctx.get());

    TSignal<void(int)> signal;
    auto Connection = slot.connect(&signal);

    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
    signal(stream.get(), 5);
    thread.interrupt();
    thread.join();
}
