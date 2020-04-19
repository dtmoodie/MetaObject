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

    auto alloc = typed0->deviceAllocator();
    ASSERT_NE(alloc, nullptr);
    auto ptr = alloc->allocate(100 * sizeof(float), sizeof(float));
    ASSERT_NE(ptr, nullptr);

    auto device_stream = typed0->getStream();

    mo::cuda::CUDAMemoryBlock mem(1);
    cudaMemsetAsync(mem.begin(), 0, 1, typed0->getStream());

    cuda_tests::wait(ct::ptrCast<const bool>(mem.begin()), typed0->getStream());

    volatile bool callback_invoked = false;
    typed0->enqueueCallback([&callback_invoked]() { callback_invoked = true; });

    ASSERT_EQ(callback_invoked, false);

    boost::this_fiber::sleep_for(std::chrono::milliseconds(10));

    ASSERT_EQ(callback_invoked, false);

    cuda_tests::set(ct::ptrCast<bool>(mem.begin()), typed1->getStream());
    while (!callback_invoked)
    {
        boost::this_fiber::yield();
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
    stream->synchronize();

    auto h_alloc = stream->hostAllocator();
    ASSERT_NE(h_alloc, nullptr);
    mo::TDynArray<float, mo::Allocator> h_data(h_alloc, num_elems);
    ASSERT_NE(h_data.view().data(), nullptr);

    cuda_tests::set(d_data.mutableView().data(), 2.0F, num_elems, typed->getStream());
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
    stream->deviceToHost(h_data.mutableView(), d_data.view());
    stream->synchronize();
    for (size_t i = 0; i < h_data.size(); ++i)
    {
        ASSERT_EQ(h_data.view()[i], 100.0F);
    }
}
