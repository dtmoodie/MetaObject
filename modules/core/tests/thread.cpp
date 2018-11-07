#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

#include <MetaObject/thread/Thread.hpp>
#include <MetaObject/thread/ThreadPool.hpp>

using namespace mo;

namespace
{
    struct Fixture
    {
        ThreadPool m_thread_pool;
        ThreadHandle m_handle;

        volatile int count = 0;

        Fixture()
            : m_handle(m_thread_pool.requestThread())
        {
        }

        void testInit()
        {
            BOOST_REQUIRE(m_handle.context());
            m_handle.setExitCallback([]() { std::cout << "Thread shutting down" << std::endl; });
        }

        void testWork()
        {
            volatile bool executed = false;
            m_handle.pushWork([&executed]() {
                std::cout << "Executing work" << std::endl;
                executed = true;
            });
            int count = 0;
            while (!executed)
            {
                boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
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
            volatile uint32_t execute_count = 0;
            volatile bool done = false;
            for (size_t i = 0; i < 10; ++i)
            {
                m_handle.pushEventQueue([&execute_count]() { ++execute_count; });
            }
            m_handle.pushEventQueue([&done]() { done = true; });
            while (!done)
            {
                boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
            }
            BOOST_REQUIRE_EQUAL(execute_count, 10);

            execute_count = 0;
            done = false;

            m_handle.pushEventQueue([]() { boost::this_thread::sleep_for(boost::chrono::milliseconds(10)); });

            for (size_t i = 0; i < 10; ++i)
            {
                m_handle.pushEventQueue([&execute_count]() { ++execute_count; }, 1);
            }

            m_handle.pushEventQueue([&done]() {
                done = true;
                ;
            });
            while (!done)
            {
                boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
            }
            BOOST_REQUIRE_EQUAL(execute_count, 1);

            done = false;

            m_handle.pushEventQueue([]() { boost::this_thread::sleep_for(boost::chrono::milliseconds(10)); });
            for (size_t i = 0; i < 10; ++i)
            {
                m_handle.pushEventQueue(this, &Fixture::increment);
            }
            m_handle.pushEventQueue([&done]() {
                done = true;
                ;
            });
            while (!done)
            {
                boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
            }

            BOOST_REQUIRE_EQUAL(count, 1);
        }

        volatile std::atomic<uint32_t> loop_count;
        volatile bool exit_loop = false;
        void loop1()
        {
            ++loop_count;
            if (!exit_loop)
            {
                boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
                m_handle.pushEventQueue(this, &Fixture::loop1);
            }
        }

        void loop2()
        {
            ++loop_count;
            if (loop_count < 100)
            {
                m_handle.pushEventQueue(this, &Fixture::loop2);
            }
        }

        void testLoop()
        {
            loop_count = 0;
            m_handle.pushEventQueue(this, &Fixture::loop1);
            boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
            exit_loop = true;
            const uint32_t count = loop_count;
            BOOST_REQUIRE_GT(count, 0);
            boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
            BOOST_REQUIRE_LT(loop_count, count + 100);

            loop_count = 0;
            m_handle.pushEventQueue(this, &Fixture::loop2);
            boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));
            BOOST_REQUIRE_EQUAL(loop_count, 100);
        }
    };
}

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
