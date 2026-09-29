#include <MetaObject/cuda/Event.hpp>
#include <MetaObject/cuda/MemoryBlock.hpp>
#include <MetaObject/cuda/Stream.hpp>
#include <MetaObject/logging/logging.hpp>

#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>

#include <boost/fiber/operations.hpp>

#include <thread>

#include <gtest/gtest.h>

#include "device_funcs.hpp"

TEST(cuda_stream_sync, two_streams)
{
    mo::cuda::Stream stream1;
    mo::cuda::Stream stream2;
    mo::cuda::CUDAMemoryBlock mem(1024);

    // Keep stream1 busy with a bounded, memory-less kernel. The indefinite
    // spin-on-a-device-flag kernel this test used to launch stalls work on
    // other streams on some drivers (e.g. RTX 5090 + 580.x), hanging the
    // test at 100% GPU utilization; a bounded kernel cannot stall the device
    // and needs no release.
    cuda_tests::hold(2000000, stream1); // ~2 s

    ASSERT_EQ(stream1.query(), false);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    ASSERT_EQ(stream1.query(), false);

    // Do real work on stream2; it must complete while the hold is still
    // running. If stream1 == stream2, the memset would be queued behind the
    // hold, so stream2.synchronize() would wait ~2 s and stream1 would
    // already be complete at the query below — thus this tests to make sure
    // two separate streams are created and used.
    cudaMemsetAsync(mem.begin(), 0xAB, mem.size(), stream2);
    stream2.synchronize();

    ASSERT_EQ(stream1.query(), false);

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
