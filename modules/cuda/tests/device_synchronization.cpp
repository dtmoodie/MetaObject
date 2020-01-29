#include <MetaObject/cuda/Event.hpp>
#include <MetaObject/cuda/MemoryBlock.hpp>
#include <MetaObject/cuda/Stream.hpp>
#include <MetaObject/logging/logging.hpp>

#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>

#include <thread>

#include <gtest/gtest.h>

#include "device_funcs.hpp"

TEST(cuda_stream_sync, two_streams)
{
    mo::cuda::Stream stream1;
    mo::cuda::Stream stream2;
    mo::cuda::CUDAMemoryBlock mem(1);
    cudaMemsetAsync(mem.begin(), 0, 1, stream1);

    cuda_tests::wait(ct::ptrCast<const bool>(mem.begin()), stream1);

    // The above kernel will wait and thus block stream1 until the set kernel is invoked
    ASSERT_EQ(stream1.query(), false);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    ASSERT_EQ(stream1.query(), false);

    // If stream1 == stream2, the set kernel will not be invoked because still waiting on wait kernel
    // Thus this tests to make sure two separate streams are created and used
    cuda_tests::set(ct::ptrCast<bool>(mem.begin()), stream2);

    // This call may not always work in case stream2 executes setKernel so fast that stream1 completes, but it is highly
    // unlikely
    ASSERT_EQ(stream1.query(), false);

    stream2.synchronize();
    stream1.synchronize();
    ASSERT_EQ(stream1.query(), true);
}

TEST(cuda_stream_sync, callback)
{
    mo::cuda::Stream stream;

    const uint32_t num_elements = 100000000;

    mo::ObjectPool<CUevent_st> event_pool(2);

    mo::TMemoryBlock<float, mo::cuda::CUDA> mem(num_elements);

    cuda_tests::fill(mem.begin(), mem.end(), 1.0F, stream);
    cuda_tests::multiply(mem.size(), mem.begin(), 2.0F, stream);
    cuda_tests::multiply(mem.size(), mem.begin(), 2.0F, stream);

    mo::cuda::Event ev(&event_pool);
    bool callback_called = false;

    ASSERT_THROW(ev.setCallback([]() {}), mo::TExceptionWithCallstack<std::runtime_error>);
    ev.record(stream);

    ev.setCallback([&callback_called]() { callback_called = true; });

    ASSERT_THROW(ev.setCallback([]() {}), mo::TExceptionWithCallstack<std::runtime_error>);

    cuda_tests::multiply(mem.size(), mem.begin(), 2.0F, stream);
    cuda_tests::multiply(mem.size(), mem.begin(), 2.0F, stream);

    ASSERT_EQ(callback_called, false);
    while (!callback_called)
    {
        boost::this_fiber::yield();
    }
}

TEST(cuda_stream_sync, event)
{
    mo::cuda::Stream stream1;
    mo::cuda::Stream stream2;

    const uint32_t num_elements = 100000000;

    mo::ObjectPool<CUevent_st> event_pool(2);

    mo::TMemoryBlock<float, mo::cuda::CUDA> mem(num_elements);

    cuda_tests::fill(mem.begin(), mem.end(), 1.0F, stream1);
    cuda_tests::multiply(mem.size(), mem.begin(), 2.0F, stream1);
    cuda_tests::multiply(mem.size(), mem.begin(), 2.0F, stream1);
    cuda_tests::multiply(mem.size(), mem.begin(), 2.0F, stream1);
    cuda_tests::multiply(mem.size(), mem.begin(), 2.0F, stream1);

    mo::cuda::Event ev(&event_pool);

    ev.record(stream1);
    stream2.waitEvent(ev);

    cuda_tests::multiply(mem.size(), mem.begin(), 0.5F, stream2);
    cuda_tests::multiply(mem.size(), mem.begin(), 0.5F, stream2);
    cuda_tests::multiply(mem.size(), mem.begin(), 0.5F, stream2);
    cuda_tests::multiply(mem.size(), mem.begin(), 0.5F, stream2);

    stream1.synchronize();
    stream2.synchronize();

    mo::TMemoryBlock<float, mo::cuda::HOST> hmem(mem.size());

    cudaMemcpy(hmem.begin(), mem.begin(), mem.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // only check the first 1000 elems, otherwise this takes literally forever
    for (uint32_t i = 0; i < 1000; ++i)
    {
        ASSERT_EQ(hmem.begin()[i], 1.0F) << "at " << i;
    }
}
