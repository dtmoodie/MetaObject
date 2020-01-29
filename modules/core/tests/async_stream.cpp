#include <MetaObject/thread/FiberScheduler.hpp>

#include <MetaObject/core/IAsyncStream.hpp>

#include <MetaObject/thread/ThreadPool.hpp>

#include <boost/fiber/all.hpp>

#include "gtest/gtest.h"

TEST(async_stream, creation)
{
    ASSERT_EQ(mo::IAsyncStream::current(), nullptr);

    auto stream = mo::IAsyncStream::create();
    ASSERT_NE(stream, nullptr);
    ASSERT_EQ(mo::IAsyncStream::current(), stream);
}

TEST(async_stream, work)
{
    auto scheduler = mo::PriorityScheduler::current();
    ASSERT_EQ(scheduler->size(), 1);

    auto stream = mo::IAsyncStream::create();

    bool work_complete = false;
    stream->pushWork([&work_complete]() { work_complete = true; });
    ASSERT_EQ(scheduler->size(), 2);
    boost::this_fiber::sleep_for(std::chrono::milliseconds(100));
    ASSERT_EQ(work_complete, true);
}

TEST(async_stream, work_sync)
{
    auto scheduler = mo::PriorityScheduler::current();
    ASSERT_EQ(scheduler->size(), 1);

    auto stream = mo::IAsyncStream::create();

    bool work_complete = false;
    stream->pushWork([&work_complete]() { work_complete = true; });
    stream->synchronize();
    ASSERT_EQ(work_complete, true);
}

TEST(async_stream, current_stream)
{
    auto scheduler = mo::PriorityScheduler::current();
    ASSERT_EQ(scheduler->size(), 1);

    auto stream = mo::IAsyncStream::create();

    bool check_passes = false;
    stream->pushWork([&check_passes, stream]() { check_passes = stream == mo::IAsyncStream::current(); });
    stream->synchronize();
    ASSERT_EQ(check_passes, true);
}
