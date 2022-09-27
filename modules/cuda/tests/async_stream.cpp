#include <MetaObject/cuda/AsyncStream.hpp>
#include <MetaObject/cuda/Event.hpp>
#include <MetaObject/cuda/MemoryBlock.hpp>
#include <MetaObject/cuda/Stream.hpp>
#include <MetaObject/thread/fiber_include.hpp>
#include <MetaObject/types/TDynArray.hpp>

#include <MetaObject/logging/profiling.hpp>

#include <MetaObject/thread/Mutex.hpp>
#include <boost/fiber/operations.hpp>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include "device_funcs.hpp"

TEST(cuda_stream, creation)
{
    auto stream = mo::IAsyncStream::create();
    auto typed = std::dynamic_pointer_cast<mo::cuda::AsyncStream>(stream);
    ASSERT_NE(typed, nullptr);
}

TEST(cuda_stream, allocate)
{
    auto stream = mo::IAsyncStream::create();
    auto typed = std::dynamic_pointer_cast<mo::cuda::AsyncStream>(stream);
    ASSERT_NE(typed, nullptr);

    auto alloc = typed->deviceAllocator();
    ASSERT_NE(alloc, nullptr);
    auto ptr = alloc->allocate(100 * sizeof(float), sizeof(float));
    ASSERT_NE(ptr, nullptr);

    alloc->deallocate(ptr, 100 * sizeof(float));
}

TEST(cuda_stream, launch)
{
    cudaFree(0);
    auto stream = mo::IAsyncStream::create();
    ASSERT_NE(stream, nullptr);
    auto typed0 = std::dynamic_pointer_cast<mo::cuda::AsyncStream>(stream);
    ASSERT_NE(typed0, nullptr);

    stream = mo::IAsyncStream::create();
    ASSERT_NE(stream, nullptr);
    auto typed1 = std::dynamic_pointer_cast<mo::cuda::AsyncStream>(stream);
    ASSERT_NE(typed1, nullptr);
    ASSERT_NE(typed0, typed1);
    ASSERT_NE(typed0->getStream(), typed1->getStream());

    auto device_stream0 = typed0->getStream();
    auto device_stream1 = typed1->getStream();
    ASSERT_NE(device_stream0, device_stream1);
    mo::cuda::CUDAMemoryBlock mem(1);
    {
        mo::ScopedProfile profile("asdf");

        cudaMemsetAsync(mem.begin(), 0, 1, device_stream0);
        // This makes steam0 wait until work is complete
        cuda_tests::wait(ct::ptrCast<const bool>(mem.begin()), device_stream0);
    }

    cudaError_t err = cudaPeekAtLastError();
    ASSERT_EQ(err, cudaSuccess) << "Unable to launch wait kernel due to " << cudaGetErrorString(err);

    // ASSERT_FALSE(device_stream0.query());

    volatile bool callback_invoked = false;
    // clang-format off
    typed0->enqueueCallback(
        [&callback_invoked](mo::IAsyncStream*)
        {
            callback_invoked = true;
        });
    // clang-format on

    ASSERT_EQ(callback_invoked, false);

    boost::this_fiber::sleep_for(std::chrono::milliseconds(10));

    ASSERT_EQ(callback_invoked, false);

    cuda_tests::set(ct::ptrCast<bool>(mem.begin()), typed1->getStream());
    while (!callback_invoked)
    {
        typed0->synchronize();
        typed1->synchronize();
    }
    ASSERT_EQ(callback_invoked, true);
}

TEST(cuda_stream, async_copy)
{
    const size_t num_elems = 10000;

    auto stream = mo::IDeviceStream::create();
    ASSERT_NE(stream, nullptr);

    auto typed = std::dynamic_pointer_cast<mo::cuda::AsyncStream>(stream);
    ASSERT_NE(typed, nullptr);

    auto d_alloc = stream->deviceAllocator();
    ASSERT_NE(d_alloc, nullptr);

    mo::TDynArray<float, mo::DeviceAllocator> d_data(d_alloc, num_elems);
    ASSERT_NE(d_data.view().data(), nullptr);
    cuda_tests::set(d_data.mutableView().data(), 1.0F, num_elems, typed->getStream());
    cudaError_t err = cudaPeekAtLastError();
    ASSERT_EQ(err, cudaSuccess) << "Unable to launch set kernel due to " << cudaGetErrorString(err);
    stream->synchronize();

    auto h_alloc = stream->hostAllocator();
    ASSERT_NE(h_alloc, nullptr);
    mo::TDynArray<float, mo::Allocator> h_data(h_alloc, num_elems);
    ASSERT_NE(h_data.view().data(), nullptr);

    cuda_tests::set(d_data.mutableView().data(), 2.0F, num_elems, typed->getStream());
    err = cudaPeekAtLastError();
    ASSERT_EQ(err, cudaSuccess) << "Unable to launch set kernel due to " << cudaGetErrorString(err);
    ASSERT_FALSE(typed->getStream().query());
    stream->deviceToHost(h_data.mutableView(), d_data.view());
    stream->synchronize();
    for (size_t i = 0; i < h_data.size(); ++i)
    {
        ASSERT_EQ(h_data.view()[i], 2.0F);
    }

    auto view = h_data.mutableView();
    for (auto& val : view)
    {
        val = 10.0F;
    }
    stream->hostToDevice(d_data.mutableView(), h_data.view());
    cuda_tests::square(d_data.mutableView().data(), d_data.size(), typed->getStream());
    err = cudaPeekAtLastError();
    ASSERT_EQ(err, cudaSuccess) << "Unable to launch square kernel due to " << cudaGetErrorString(err);
    stream->deviceToHost(h_data.mutableView(), d_data.view());
    stream->synchronize();
    for (size_t i = 0; i < h_data.size(); ++i)
    {
        ASSERT_EQ(h_data.view()[i], 100.0F);
    }
}
