#include <MetaObject/thread/fiber_include.hpp>

#include <MetaObject/core/SystemTable.hpp>
#include <MetaObject/thread/FiberScheduler.hpp>
#include <MetaObject/thread/Thread.hpp>
#include <MetaObject/thread/ThreadPool.hpp>
#include <MetaObject/thread/ThreadRegistry.hpp>

#include <boost/fiber/fiber.hpp>
#include <boost/fiber/operations.hpp>

#include "gtest/gtest.h"

#include <iostream>

using namespace mo;

/*
Things that we want to test:
- pushing work to a stream on the same thread
- pushing work to a stream on a different thread
- stream prioritization
- events and overriding of events
- helper thread spawning and helping to process tasks

*/

namespace
{
    struct RawFiberFixture : ::testing::Test
    {
        volatile int count = 0;
        RawFiberFixture()
        {
            auto pool = mo::singleton<mo::ThreadPool>();
            auto schedulers = pool->getSchedulers();
            EXPECT_EQ(schedulers.size(), 1);
        }

        void testWork()
        {
            volatile bool executed = false;
            boost::fibers::fiber fiber([&executed]() {
                std::cout << "Executing work" << std::endl;
                executed = true;
            });
            fiber.detach();
            int count = 0;
            while (!executed)
            {
                boost::this_fiber::sleep_for(1 * ms);
                ++count;
                ASSERT_LT(count, 100);
            }
        }

        void increment()
        {
            ++count;
        }

        void testEvent()
        {
        }

        volatile std::atomic<uint32_t> loop_count;
        volatile bool exit_loop = false;
        void loop1()
        {
            ++loop_count;
            if (!exit_loop)
            {
                boost::this_fiber::sleep_for(1 * ms);
                boost::fibers::fiber fiber([this]() { loop1(); });
                fiber.detach();
            }
        }

        void loop2()
        {
            ++loop_count;
            if (loop_count < 100)
            {
                boost::this_fiber::sleep_for(1 * ms);
                boost::fibers::fiber fiber([this]() { loop2(); });
                fiber.detach();
            }
        }

        void testLoop()
        {
            loop_count = 0;
            boost::fibers::fiber fiber([this]() { loop1(); });
            fiber.detach();
            boost::this_fiber::sleep_for(100 * ms);
            exit_loop = true;
            const uint32_t count = loop_count;
            ASSERT_GT(count, 0);
            boost::this_fiber::sleep_for(100 * ms);
            ASSERT_LT(loop_count, count + 100);

            loop_count = 0;

            boost::this_fiber::sleep_for(1000 * ms);
            ASSERT_EQ(loop_count, 0);

            loop2();

            boost::this_fiber::sleep_for(1000 * ms);
            ASSERT_EQ(loop_count, 100);
        }
    };

    struct StreamFixture : ::testing::Test
    {
        IAsyncStreamPtr_t m_stream;
        IAsyncStreamPtr_t m_high_stream;

        StreamFixture()
            : execution_count(0)
        {
            m_stream = AsyncStream::create();
            m_high_stream = AsyncStream::create();
            m_high_stream->setHostPriority(PriorityLevels::HIGHEST);
        }

        int count = 0;
        bool stop = false;
        void loopImpl()
        {
            ++count;
            if (!stop)
            {
                boost::this_fiber::sleep_for(1 * ms);
                m_stream->pushWork([this](mo::IAsyncStream*) { loopImpl(); });
            }
        }

        void testLoop()
        {
            stop = false;
            count = 0;
            m_stream->pushWork([this](mo::IAsyncStream*) { loopImpl(); });

            boost::this_fiber::sleep_for(100 * ms);
            stop = true;
            boost::this_fiber::sleep_for(10 * ms);
            ASSERT_GT(count, 80);
        }

        void testEventComplex()
        {
            execution_count = 0;
            for (uint32_t i = 0; i < 5; ++i)
            {
                m_stream->pushEvent([](mo::IAsyncStream*) {}, i);
            }

            m_stream->pushEvent([this](mo::IAsyncStream*) { ++execution_count; }, 15);

            m_stream->pushEvent([this](mo::IAsyncStream*) { ++execution_count; }, 15);

            boost::this_fiber::sleep_for(1 * ms);
            ASSERT_EQ(execution_count, 1);
        }

        volatile std::atomic<uint32_t> execution_count;

        void testSpawningOfAssistant()
        {
            auto pool = mo::singleton<mo::ThreadPool>();
            auto schedulers = pool->getSchedulers();
            ASSERT_EQ(schedulers.size(), 1);

            execution_count = 0;

            for (uint32_t i = 0; i < 1000; ++i)
            {
                m_stream->pushWork([this](mo::IAsyncStream*) {
                    ++execution_count;
                    boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
                });
                boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
            }
            uint32_t count = execution_count;
            schedulers = pool->getSchedulers();
            ASSERT_EQ(schedulers.size(), 2);
            ASSERT_GT(count, 0);
            boost::this_fiber::sleep_for(2 * second);
            count = execution_count;
            ASSERT_EQ(count, 1000);
        }
    };
} // namespace

TEST_F(RawFiberFixture, work)
{
    testWork();
}

TEST_F(RawFiberFixture, event)
{
    testEvent();
}

TEST_F(RawFiberFixture, loop)
{
    testLoop();
}

TEST_F(StreamFixture, stream_loop)
{
    testLoop();
}

// This whole test is kinda invalid now that we have a new approach to fibers and streams
/*TEST_F(StreamFixture, priority)
{
    bool higher_priority_executed = false;
    bool lower_priority_executed = false;
    auto pool = mo::singleton<mo::ThreadPool>();
    auto schedulers = pool->getSchedulers();
    m_stream->pushWork([&lower_priority_executed, &higher_priority_executed](mo::IAsyncStream*) {
        lower_priority_executed = true;
        ASSERT_EQ(higher_priority_executed, true);
    });

    m_high_stream->pushWork([&lower_priority_executed, &higher_priority_executed](mo::IAsyncStream*) {
        higher_priority_executed = true;
        ASSERT_EQ(lower_priority_executed, false);
    });


    ASSERT_TRUE(higher_priority_executed);
    ASSERT_TRUE(lower_priority_executed);
}*/

TEST_F(StreamFixture, stream_event_simple)
{
    execution_count = 0;
    bool first_executed = false;
    bool second_executed = false;

    m_stream->pushEvent(
        [this, &first_executed](mo::IAsyncStream*) {
            ++execution_count;
            first_executed = true;
        },
        15);

    m_stream->pushEvent(
        [this, &second_executed](mo::IAsyncStream*) {
            ++execution_count;
            second_executed = true;
        },
        15);

    m_stream->synchronize();
    EXPECT_EQ(execution_count, 1);
    EXPECT_FALSE(first_executed);
    EXPECT_TRUE(second_executed);
}

/*TEST_F(StreamFixture, spawn_assistant)
{
    testSpawningOfAssistant();
    auto pool = mo::singleton<mo::ThreadPool>();
    auto schedulers = pool->getSchedulers();
    //schedulers[0]->releaseAssistant();
    pool->cleanup();
    schedulers = pool->getSchedulers();
    ASSERT_EQ(schedulers.size(), 1);
}*/

TEST(stream, execute_fiber_with_yield)
{
    bool callback_called = false;
    boost::fibers::fiber fib([&callback_called]() { callback_called = true; });
    fib.detach();
    ASSERT_FALSE(callback_called);

    boost::this_fiber::yield();
    ASSERT_TRUE(callback_called);
}

// So the problem that I'm having is that the work stream for the created stream is being associated with the wrong
// thread
TEST(worker_thread, execute_on_worker)
{
    auto worker = mo::ThreadRegistry::instance()->getThread(mo::ThreadRegistry::WORKER);
    const size_t this_thread = mo::getThisThread();
    auto work = [this_thread](mo::IAsyncStream*) {
        const size_t work_thread = mo::getThisThread();
        ASSERT_NE(this_thread, work_thread);
    };
    worker->pushWork(std::move(work));
    worker->synchronize();
}
