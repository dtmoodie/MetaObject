#include <boost/fiber/all.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

#include <MetaObject/core/SystemTable.hpp>
#include <MetaObject/thread/Thread.hpp>
#include <MetaObject/thread/ThreadPool.hpp>

#include <iostream>

using namespace mo;

namespace
{
    struct RawFiberFixture
    {
        volatile int count = 0;
        RawFiberFixture()
        {
            mo::ThreadPool* pool = mo::singleton<mo::ThreadPool>();
            auto schedulers = pool->getSchedulers();
            BOOST_REQUIRE_EQUAL(schedulers.size(), 1);
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
                BOOST_REQUIRE(count < 100);
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
            BOOST_REQUIRE_GT(count, 0);
            boost::this_fiber::sleep_for(100 * ms);
            BOOST_REQUIRE_LT(loop_count, count + 100);

            loop_count = 0;

            boost::this_fiber::sleep_for(1000 * ms);
            BOOST_REQUIRE_EQUAL(loop_count, 0);

            loop2();

            boost::this_fiber::sleep_for(1000 * ms);
            BOOST_REQUIRE_EQUAL(loop_count, 100);
        }
    };

    struct StreamFixture
    {
        IAsyncStreamPtr_t m_stream;

        StreamFixture()
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
            BOOST_REQUIRE_GT(count, 80);
        }

        void testEventSimple()
        {
            execution_count = 0;
            m_stream->pushEvent([this]() { ++execution_count; }, 15);

            m_stream->pushEvent([this]() { ++execution_count; }, 15);

            boost::this_fiber::sleep_for(1 * ms);
            BOOST_REQUIRE_EQUAL(execution_count, 1);
        }

        void testEventComplex()
        {
            execution_count = 0;
            for (uint32_t i = 0; i < 5; ++i)
            {
                m_stream->pushEvent([this]() {}, i);
            }

            m_stream->pushEvent([this]() { ++execution_count; }, 15);

            m_stream->pushEvent([this]() { ++execution_count; }, 15);

            boost::this_fiber::sleep_for(1 * ms);
            BOOST_REQUIRE_EQUAL(execution_count, 1);
        }

        void testWorkPriority()
        {
            bool higher_priority_executed = false;
            bool lower_priority_executed = false;
            mo::ThreadPool* pool = mo::singleton<mo::ThreadPool>();
            auto schedulers = pool->getSchedulers();
            m_stream->pushWork([&lower_priority_executed, &higher_priority_executed]() {
                lower_priority_executed = true;
                BOOST_REQUIRE(higher_priority_executed);
            });

            m_stream->pushWork(
                [&lower_priority_executed, &higher_priority_executed]() {
                    higher_priority_executed = true;
                    BOOST_REQUIRE(lower_priority_executed == false);
                },
                HIGHEST);
            boost::this_fiber::sleep_for(1 * ms);

            BOOST_REQUIRE(higher_priority_executed);
            BOOST_REQUIRE(lower_priority_executed);
        }

        volatile std::atomic<uint32_t> execution_count;

        void testSpawningOfAssistant()
        {
            mo::ThreadPool* pool = mo::singleton<mo::ThreadPool>();
            auto schedulers = pool->getSchedulers();
            BOOST_REQUIRE_EQUAL(schedulers.size(), 1);

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
            BOOST_REQUIRE_EQUAL(schedulers.size(), 2);
            BOOST_REQUIRE_GT(count, 0);
            boost::this_fiber::sleep_for(2 * second);
            count = execution_count;
            BOOST_REQUIRE_EQUAL(count, 1000);
        }
    };
}

BOOST_AUTO_TEST_SUITE(threading_tests)

BOOST_FIXTURE_TEST_CASE(work, RawFiberFixture)
{
    testWork();
}

BOOST_FIXTURE_TEST_CASE(event, RawFiberFixture)
{
    testEvent();
}

BOOST_FIXTURE_TEST_CASE(loop, RawFiberFixture)
{
    testLoop();
}

BOOST_FIXTURE_TEST_CASE(stream_loop, StreamFixture)
{
    testLoop();
}

BOOST_FIXTURE_TEST_CASE(priority, StreamFixture)
{
    testWorkPriority();
}

BOOST_FIXTURE_TEST_CASE(stream_event_simple, StreamFixture)
{
    testEventSimple();
}

BOOST_FIXTURE_TEST_CASE(spawn_assistant, StreamFixture)
{
    testSpawningOfAssistant();
}
}
