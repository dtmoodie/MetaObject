#include <boost/fiber/all.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

#include <MetaObject/thread/Thread.hpp>
#include <MetaObject/thread/ThreadPool.hpp>

#include <iostream>

using namespace mo;

namespace
{
    struct Fixture
    {
        ThreadPool m_thread_pool;
        std::shared_ptr<Thread> m_thread;
        IAsyncStreamPtr_t m_stream;
        volatile int count = 0;

        Fixture()
        {

        }

        void testInit()
        {
            m_thread = m_thread_pool.requestThread();
            BOOST_REQUIRE(m_thread);
            m_stream = m_thread->asyncStream();
            BOOST_REQUIRE(m_stream);
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
                boost::fibers::fiber fiber([this]()
                {
                    loop1();
                });
                fiber.detach();
            }
        }

        void loop2()
        {
            ++loop_count;
            if (loop_count < 100)
            {
                boost::this_fiber::sleep_for(1 * ms);
                boost::fibers::fiber fiber([this]()
                {
                    loop2();
                });
                fiber.detach();
            }
        }

        void testLoop()
        {
            loop_count = 0;
            boost::fibers::fiber fiber([this]()
            {
                loop1();
            });
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
}

BOOST_AUTO_TEST_SUITE(threading_tests)

BOOST_FIXTURE_TEST_CASE(init, Fixture)
{
    testInit();
}

BOOST_FIXTURE_TEST_CASE(work, Fixture)
{
    testWork();
}

BOOST_FIXTURE_TEST_CASE(event, Fixture)
{
    testEvent();
}

BOOST_FIXTURE_TEST_CASE(loop, Fixture)
{
    testLoop();
}
}
