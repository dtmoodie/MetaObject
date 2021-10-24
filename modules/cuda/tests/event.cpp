#include <MetaObject/cuda/AsyncStream.hpp>
#include <MetaObject/cuda/Event.hpp>
#include <MetaObject/cuda/MemoryBlock.hpp>
#include <MetaObject/cuda/Stream.hpp>
#include <MetaObject/thread/fiber_include.hpp>
#include <MetaObject/types/TDynArray.hpp>

#include <MetaObject/thread/Mutex.hpp>
#include <boost/fiber/operations.hpp>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include "device_funcs.hpp"

TEST(cuda_event, creation)
{
    std::shared_ptr<mo::ObjectPool<CUevent_st>> event_pool = mo::ObjectPool<CUevent_st>::create();
    ASSERT_NE(event_pool, nullptr);

    mo::cuda::Event event(event_pool);
}

TEST(cuda_event, query_null)
{
    std::shared_ptr<mo::ObjectPool<CUevent_st>> event_pool = mo::ObjectPool<CUevent_st>::create();
    ASSERT_NE(event_pool, nullptr);

    mo::cuda::Event event(event_pool);

    ASSERT_TRUE(event.queryCompletion());
}

TEST(cuda_event, synchronize_null)
{
    std::shared_ptr<mo::ObjectPool<CUevent_st>> event_pool = mo::ObjectPool<CUevent_st>::create();
    ASSERT_NE(event_pool, nullptr);

    mo::cuda::Event event(event_pool);

    ASSERT_TRUE(event.synchronize());
}

TEST(cuda_event, query_async_operation)
{
    std::shared_ptr<mo::ObjectPool<CUevent_st>> event_pool = mo::ObjectPool<CUevent_st>::create(1);
    mo::cuda::Event event(event_pool);
    mo::cuda::Stream stream0;
    // need a second stream for setting the value
    mo::cuda::Stream stream1;

    mo::cuda::CUDAMemoryBlock mem(1);
    cudaMemsetAsync(mem.begin(), 0, 1, stream0);

    cuda_tests::wait(ct::ptrCast<const bool>(mem.begin()), stream0);

    event.record(stream0);

    ASSERT_FALSE(event.queryCompletion());

    ASSERT_FALSE(event.synchronize(std::chrono::milliseconds(10)));

    cuda_tests::set(ct::ptrCast<bool>(mem.begin()), stream1);
    ASSERT_TRUE(event.synchronize(std::chrono::milliseconds(10)));
    ASSERT_TRUE(event.queryCompletion());
}

TEST(cuda_event, callback_from_synchronization)
{
    std::shared_ptr<mo::ObjectPool<CUevent_st>> event_pool = mo::ObjectPool<CUevent_st>::create(1);
    mo::cuda::Event event(event_pool);
    mo::cuda::Stream stream0;
    mo::cuda::Stream stream1;

    mo::cuda::CUDAMemoryBlock mem(1);
    cudaMemsetAsync(mem.begin(), 0, 1, stream0);
    cuda_tests::wait(ct::ptrCast<const bool>(mem.begin()), stream0);

    event.record(stream0);

    ASSERT_FALSE(event.queryCompletion());
    ASSERT_FALSE(event.synchronize(std::chrono::milliseconds(10)));
    bool callback_called = false;
    event.setCallback([&callback_called](mo::IAsyncStream*) { callback_called = true; });

    ASSERT_FALSE(callback_called);

    cuda_tests::set(ct::ptrCast<bool>(mem.begin()), stream1);
    ASSERT_TRUE(event.synchronize(std::chrono::milliseconds(10)));
    ASSERT_TRUE(callback_called);
}

TEST(cuda_event, callback_from_yield)
{
    std::shared_ptr<mo::ObjectPool<CUevent_st>> event_pool = mo::ObjectPool<CUevent_st>::create(1);
    mo::cuda::Event event(event_pool);
    mo::cuda::Stream stream0;
    mo::cuda::Stream stream1;

    mo::cuda::CUDAMemoryBlock mem(1);
    cudaMemsetAsync(mem.begin(), 0, 1, stream0);
    cuda_tests::wait(ct::ptrCast<const bool>(mem.begin()), stream0);

    event.record(stream0);

    ASSERT_FALSE(event.queryCompletion());
    ASSERT_FALSE(event.synchronize(std::chrono::milliseconds(10)));
    bool callback_called = false;
    event.setCallback([&callback_called](mo::IAsyncStream*) { callback_called = true; });

    ASSERT_FALSE(callback_called);

    cuda_tests::set(ct::ptrCast<bool>(mem.begin()), stream1);
    boost::this_fiber::sleep_for(std::chrono::milliseconds(100));
    ASSERT_TRUE(callback_called);
}
