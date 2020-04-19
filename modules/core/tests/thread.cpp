#include <MetaObject/thread/fiber_include.hpp>

#include <MetaObject/core/SystemTable.hpp>
#include <MetaObject/thread/FiberScheduler.hpp>
#include <MetaObject/thread/Thread.hpp>
#include <MetaObject/thread/ThreadPool.hpp>

#include <boost/fiber/fiber.hpp>
#include <boost/fiber/operations.hpp>

#include "gtest/gtest.h"

#include <iostream>

using namespace mo;

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

        StreamFixture()
            : execution_count(0)
        {
            m_stream = std::make_shared<AsyncStream>();
        }

        int count = 0;
        bool stop = false;
        void loopImpl()
        {
            ++count;
            if (!stop)
            {
                boost::this_fiber::sleep_for(1 * ms);
                m_stream->pushWork([this]() { loopImpl(); });
            }
        }

        void testLoop()
        {
            stop = false;
            count = 0;
            m_stream->pushWork([this]() { loopImpl(); });

            boost::this_fiber::sleep_for(100 * ms);
            stop = true;
            boost::this_fiber::sleep_for(10 * ms);
            ASSERT_GT(count, 80);
        }

        void testEventSimple()
        {
            execution_count = 0;
            m_stream->pushEvent([this]() { ++execution_count; }, 15);

            m_stream->pushEvent([this]() { ++execution_count; }, 15);

            boost::this_fiber::sleep_for(1 * ms);
            ASSERT_EQ(execution_count, 1);
        }

        void testEventComplex()
        {
            execution_count = 0;
            for (uint32_t i = 0; i < 5; ++i)
            {
                m_stream->pushEvent([]() {}, i);
            }

            m_stream->pushEvent([this]() { ++execution_count; }, 15);

            m_stream->pushEvent([this]() { ++execution_count; }, 15);

            boost::this_fiber::sleep_for(1 * ms);
            ASSERT_EQ(execution_count, 1);
        }

        void testWorkPriority()
        {
            bool higher_priority_executed = false;
            bool lower_priority_executed = false;
            auto pool = mo::singleton<mo::ThreadPool>();
            auto schedulers = pool->getSchedulers();
            m_stream->pushWork([&lower_priority_executed, &higher_priority_executed]() {
                lower_priority_executed = true;
                ASSERT_EQ(higher_priority_executed, true);
            });

            m_stream->pushWork(
                [&lower_priority_executed, &higher_priority_executed]() {
                    higher_priority_executed = true;
                    ASSERT_EQ(lower_priority_executed, false);
                },
                HIGHEST);
            boost::this_fiber::sleep_for(1 * ms);

            ASSERT_EQ(higher_priority_executed, true);
            ASSERT_EQ(lower_priority_executed, true);
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
                m_stream->pushWork([this]() {
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

TEST_F(StreamFixture, priority)
{
    testWorkPriority();
}

TEST_F(StreamFixture, stream_event_simple)
{
    testEventSimple();
}

TEST_F(StreamFixture, spawn_assistant)
{
    testSpawningOfAssistant();
    auto pool = mo::singleton<mo::ThreadPool>();
    auto schedulers = pool->getSchedulers();
    schedulers[0]->releaseAssistant();
    pool->cleanup();
    schedulers = pool->getSchedulers();
    ASSERT_EQ(schedulers.size(), 1);
}
