#include <MetaObject/cuda/Event.hpp>
#include <MetaObject/cuda/Stream.hpp>
#include <MetaObject/cuda/MemoryBlock.hpp>

#include <MetaObject/thread/fiber_include.hpp>

#include <boost/test/test_tools.hpp>
#include <boost/test/auto_unit_test.hpp>

#include <MetaObject/thread/fiber_include.hpp>

#include <thrust/fill.h>
#include <thrust/device_ptr.h>

#include <host_defines.h>
#include <cuda_runtime_api.h>

#include <thread>

uint32_t divUp(const uint32_t total, const uint32_t threads)
{
    return (total + threads - 1 ) / threads;
}

__global__
void waitKernel(volatile const bool* quit)
{
    while(!*quit)
    {

    }
}

__global__
void setKernel(volatile bool* quit)
{
    *quit = true;
}

__global__
void mulKernel(const uint32_t N, float* data, const float scale)
{
    const uint32_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    if(thread_index < N)
    {
        data[thread_index] *= scale;
    }
}


BOOST_AUTO_TEST_CASE(two_streams)
{
    mo::cuda::Stream stream1;
    mo::cuda::Stream stream2;
    mo::cuda::CUDAMemoryBlock mem(1);
    cudaMemsetAsync(mem.begin(), 0, 1, stream1);

    waitKernel<<<1,1,0, stream1>>>(reinterpret_cast<const bool*>(mem.begin()));

    BOOST_REQUIRE(stream1.query() == false);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    BOOST_REQUIRE(stream1.query() == false);

    setKernel<<<1, 1, 0, stream2>>>(reinterpret_cast<bool*>(mem.begin()));
    BOOST_REQUIRE(stream1.query() == false);

    stream2.synchronize();
    BOOST_REQUIRE(stream1.query() == true);
}

BOOST_AUTO_TEST_CASE(stream_callback)
{
    mo::cuda::Stream stream;

    const uint32_t num_elements = 100000000;
    const uint32_t num_threads = 256;
    const uint32_t num_blocks = divUp(num_elements, num_threads);

    mo::ObjectPool<CUevent_st> event_pool(2);


    mo::TMemoryBlock<float, mo::cuda::CUDA> mem(num_elements);

    thrust::fill(thrust::device_ptr<float>(mem.begin()), thrust::device_ptr<float>(mem.end()), 1.0F);

    mulKernel<<<num_blocks,num_threads, 0, stream>>>(mem.size(), mem.begin(), 2.0F);
    mulKernel<<<num_blocks,num_threads, 0, stream>>>(mem.size(), mem.begin(), 2.0F);

    mo::cuda::Event ev(&event_pool);
    bool callback_called = false;

    BOOST_CHECK_THROW(ev.setCallback([](){}), std::runtime_error);
    ev.record(stream);

    ev.setCallback([&callback_called]()
    {
        callback_called = true;
    });

    BOOST_CHECK_THROW(ev.setCallback([](){}), std::runtime_error);

    mulKernel<<<num_blocks,num_threads, 0, stream>>>(mem.size(), mem.begin(), 2.0F);
    mulKernel<<<num_blocks,num_threads, 0, stream>>>(mem.size(), mem.begin(), 2.0F);

    BOOST_REQUIRE(callback_called == false);
    while(callback_called == false)
    {
        boost::this_fiber::yield();
    }
}

BOOST_AUTO_TEST_CASE(stream_sync_event)
{
    mo::cuda::Stream stream1;
    mo::cuda::Stream stream2;

    const uint32_t num_elements = 100000000;
    const uint32_t num_threads = 256;
    const uint32_t num_blocks = divUp(num_elements, num_threads);

    mo::ObjectPool<CUevent_st> event_pool(2);

    mo::TMemoryBlock<float, mo::cuda::CUDA> mem(num_elements);

    thrust::fill(thrust::device_ptr<float>(mem.begin()), thrust::device_ptr<float>(mem.end()), 1.0F);

    mulKernel<<<num_blocks,num_threads, 0, stream1>>>(mem.size(), mem.begin(), 2.0F);
    mulKernel<<<num_blocks,num_threads, 0, stream1>>>(mem.size(), mem.begin(), 2.0F);
    mulKernel<<<num_blocks,num_threads, 0, stream1>>>(mem.size(), mem.begin(), 2.0F);
    mulKernel<<<num_blocks,num_threads, 0, stream1>>>(mem.size(), mem.begin(), 2.0F);

    mo::cuda::Event ev(&event_pool);

    ev.record(stream1);
    stream2.waitEvent(ev);

    mulKernel<<<num_blocks,num_threads,0, stream2>>>(mem.size(), mem.begin(), 0.5F);
    mulKernel<<<num_blocks,num_threads,0, stream2>>>(mem.size(), mem.begin(), 0.5F);
    mulKernel<<<num_blocks,num_threads,0, stream2>>>(mem.size(), mem.begin(), 0.5F);
    mulKernel<<<num_blocks,num_threads,0, stream2>>>(mem.size(), mem.begin(), 0.5F);
    stream1.synchronize();
    stream2.synchronize();

    mo::TMemoryBlock<float, mo::cuda::HOST> hmem(mem.size());

    cudaMemcpy(hmem.begin(), mem.begin(), mem.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // only check the first 1000 elems, otherwise this takes literally forever
    for(uint32_t i = 0; i < 1000; ++i)
    {
        BOOST_REQUIRE_EQUAL(hmem.begin()[i], 1.0F);
    }

}
