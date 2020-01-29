#ifdef MO_HAVE_OPENCV
#include <MetaObject/core/detail/Allocator.hpp>
#include <MetaObject/core/detail/StlAllocator.hpp>
#include <MetaObject/core/detail/Time.hpp>

#include <MetaObject/logging/logging.hpp>
#include <MetaObject/logging/profiling.hpp>

#include <MetaObject/core/detail/allocator_policies/Combined.hpp>
#include <MetaObject/core/detail/allocator_policies/Continuous.hpp>
#include <MetaObject/core/detail/allocator_policies/Lock.hpp>
#include <MetaObject/core/detail/allocator_policies/Pool.hpp>
#include <MetaObject/core/detail/allocator_policies/Stack.hpp>
#include <MetaObject/core/detail/allocator_policies/opencv.hpp>

#include <MetaObject/cuda/MemoryBlock.hpp>
#include <MetaObject/cuda/opencv.hpp>

#include <opencv2/cudaarithm.hpp>

#include <gtest/gtest.h>

using namespace mo;
namespace
{
    struct RestoreAllocator
    {
        RestoreAllocator()
        {
            alloc = cv::cuda::GpuMat::defaultAllocator();
        }

        ~RestoreAllocator()
        {
            cv::cuda::GpuMat::setDefaultAllocator(alloc);
        }
        cv::cuda::GpuMat::Allocator* alloc;
    };
}

TEST(gpu_allocator, random_pattern)
{
    RestoreAllocator restore;
    cv::cuda::Stream stream;

    cv::cuda::GpuMat X_(1, 1000, CV_32F);
    cv::cuda::GpuMat Y_(1, 1000, CV_32F);
    auto start = mo::Time::now();
    for (int i = 0; i < 1000; ++i)
    {
        int cols = std::min(1000, 1 + rand());
        cv::cuda::GpuMat X = X_.colRange(0, cols);
        cv::cuda::GpuMat Y = Y_.colRange(0, cols);
        cv::cuda::multiply(X, cv::Scalar(100), Y, 1, -1, stream);
        cv::cuda::subtract(Y, cv::Scalar(100), Y, cv::noArray(), -1, stream);
    }
    auto end = mo::Time::now();
    double zero_allocator = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Default allocator
    start = mo::Time::now();
    for (int i = 0; i < 1000; ++i)
    {
        cv::cuda::GpuMat X(1, std::min(1000, 1 + rand()), CV_32F);
        cv::cuda::GpuMat Y;
        cv::cuda::multiply(X, cv::Scalar(100), Y, 1, -1, stream);
        cv::cuda::subtract(Y, cv::Scalar(100), Y, cv::noArray(), -1, stream);
    }
    end = mo::Time::now();
    double default_allocator_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    start = mo::Time::now();
    CombinedPolicy<PoolPolicy<cuda::CUDA>, StackPolicy<cuda::CUDA>> combined_allocator;
    mo::cuda::AllocatorProxy<> combined_allocator_proxy(&combined_allocator);
    cv::cuda::GpuMat::setDefaultAllocator(&combined_allocator_proxy);
    for (int i = 0; i < 1000; ++i)
    {
        cv::cuda::GpuMat X(1, std::min(1000, 1 + rand()), CV_32F);
        cv::cuda::GpuMat Y;
        cv::cuda::multiply(X, cv::Scalar(100), Y, 1, -1, stream);
        cv::cuda::subtract(Y, cv::Scalar(100), Y, cv::noArray(), -1, stream);
    }

    end = mo::Time::now();
    const auto pool_allocator = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "\n ======================= GPU ============================================ \n";
    std::cout << "------------ Thread specifc ---------------\n"
              << " Zero Allocation:   " << zero_allocator << "\n"
              << " Default Allocator: " << default_allocator_time << "\n"
              << " Pool Allocator:    " << pool_allocator << "\n";
}

TEST(gpu_allocator, static_allocation_pattern)
{
    cv::cuda::Stream stream;
    RestoreAllocator restore;
    // Manual buffer control
    auto start = mo::Time::now();
    {
        cv::cuda::GpuMat X(2000, 2000, CV_32F);
        cv::cuda::GpuMat Y(2000, 2000, CV_32F);
        for (int i = 0; i < 1000; ++i)
        {
            cv::cuda::multiply(X, cv::Scalar(100), Y, 1, -1, stream);
            cv::cuda::subtract(Y, cv::Scalar(100), Y, cv::noArray(), -1, stream);
        }
    }

    auto end = mo::Time::now();
    const auto zero_allocation_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Default allocator
    start = mo::Time::now();
    for (int i = 0; i < 1000; ++i)
    {
        cv::cuda::GpuMat X(2000, 2000, CV_32F);
        cv::cuda::GpuMat Y;
        cv::cuda::multiply(X, cv::Scalar(100), Y, 1, -1, stream);
        cv::cuda::subtract(Y, cv::Scalar(100), Y, cv::noArray(), -1, stream);
    }
    end = mo::Time::now();
    const auto default_allocator_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Custom allocator
    PoolPolicy<cuda::CUDA> pool_allocator;
    mo::cuda::AllocatorProxy<> pool_allocator_proxy(&pool_allocator);

    cv::cuda::GpuMat::setDefaultAllocator(&pool_allocator_proxy);
    start = mo::Time::now();
    for (int i = 0; i < 1000; ++i)
    {
        cv::cuda::GpuMat X(2000, 2000, CV_32F);
        cv::cuda::GpuMat Y;
        cv::cuda::multiply(X, cv::Scalar(100), Y, 1, -1, stream);
        cv::cuda::subtract(Y, cv::Scalar(100), Y, cv::noArray(), -1, stream);
    }

    end = mo::Time::now();
    const auto pool_allocator_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "\n ======================= GPU ============================================ \n";
    std::cout << "------------ Thread specifc ---------------\n"
              << " Zero Allocation:   " << zero_allocation_time << "\n"
              << " Default Allocator: " << default_allocator_time << "\n"
              << " Pool Allocator:    " << pool_allocator_time << "\n";
}

/*BOOST_AUTO_TEST_CASE(stl_allocator_pool)
{
    std::vector<float> zero_allocation;
    zero_allocation.resize(2000);
    auto start = mo::Time::now();
    for (int i = 0; i < 10000; ++i)
    {
        int size = std::min(2000, rand() + 1);
        cv::Mat view(1, size, CV_32F, zero_allocation.data());
        view *= 100;
        view += 10;
    }
    auto end = mo::Time::now();
    double zero_allocation_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    start = mo::Time::now();
    for (int i = 0; i < 10000; ++i)
    {
        std::vector<float> vec;
        int size = std::min(2000, rand() + 1);
        vec.resize(size);
        cv::Mat view(1, size, CV_32F, vec.data());
        view *= 100;
        view += 10;
    }
    end = mo::Time::now();
    double default_allocation = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    start = mo::Time::now();
#ifndef _MSC_VER
    for (int i = 0; i < 10000; ++i)
    {
        std::vector<float, PinnedStlAllocator<float>> vec;
        int size = std::min(2000, rand() + 1);
        vec.resize(size);
        cv::Mat view(1, size, CV_32F, vec.data());
        view *= 100;
        view += 10;
    }
#endif
    end = mo::Time::now();
    double pinned_allocation = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    start = mo::Time::now();
#ifndef _MSC_VER
    for (int i = 0; i < 10000; ++i)
    {
        std::vector<float, PinnedStlAllocatorPoolThread<float>> vec;
        int size = std::min(2000, rand() + 1);
        vec.resize(size);
        cv::Mat view(1, size, CV_32F, vec.data());
        view *= 100;
        view += 10;
    }
#endif
    end = mo::Time::now();
    double pooled_allocation = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "\n ======================= STL ============================================ \n";
    std::cout << "Zero Allocation:    " << zero_allocation_time << "\n"
              << "Default Allocation: " << default_allocation << "\n"
              << "Pinned Allocation:  " << pinned_allocation << "\n"
              << "Pooled Allocation:  " << pooled_allocation << "\n";
}*/

TEST(gpu_allocator, async_transfer_rate_random)
{
    RestoreAllocator restore;
    mo::Time start, end;
    double zero_allocation_time, default_time, pooled_time;
    const int num_iterations = 500;
    std::vector<int> sizes;
    sizes.resize(num_iterations);
    std::cout << "\n ======================= H2D -> D2H ============================================ \n";
    for (int i = 0; i < num_iterations; ++i)
    {
        int size = std::min(1000, rand() + 1);
        sizes[i] = size;
    }
    {

        std::vector<cv::Mat> h_data;
        std::vector<cv::cuda::GpuMat> d_data;
        std::vector<cv::Mat> result;
        h_data.resize(num_iterations);
        d_data.resize(num_iterations);
        result.resize(num_iterations);
        // Pre allocate
        for (int i = 0; i < num_iterations; ++i)
        {
            h_data[i] = cv::Mat(10, sizes[i], CV_32F);
            cv::cuda::createContinuous(10, sizes[i], CV_32F, d_data[i]);
            result[i] = cv::Mat(10, sizes[i], CV_32F);
        }
        cv::cuda::Stream stream;
        start = mo::Time::now();
        mo::ScopedProfile profile("zero allocation");
        for (int i = 0; i < num_iterations; ++i)
        {
            d_data[i].upload(h_data[i], stream);
            cv::cuda::multiply(d_data[i], cv::Scalar(100), d_data[i], 1, -1, stream);
            cv::cuda::add(d_data[i], cv::Scalar(100), d_data[i], cv::noArray(), -1, stream);
            d_data[i].download(result[i], stream);
        }
        end = mo::Time::now();
        zero_allocation_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Zero Allocation:    " << zero_allocation_time << "\n";
    }

    {

        cv::cuda::Stream stream;
        start = mo::Time::now();
        mo::ScopedProfile profile("default allocation");
        for (int i = 0; i < sizes.size(); ++i)
        {
            cv::Mat h_data(10, sizes[i], CV_32F);
            cv::cuda::GpuMat d_data;
            d_data.upload(h_data, stream);
            cv::cuda::multiply(d_data, cv::Scalar(100), d_data, 1, -1, stream);
            cv::cuda::add(d_data, cv::Scalar(100), d_data, cv::noArray(), -1, stream);
            cv::Mat result;
            d_data.download(result, stream);
        }
        end = mo::Time::now();
        default_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Default Allocation: " << default_time << "\n";
    }

    {
        // mo::ConcreteAllocator<mo::CpuStackPolicy, mo::StackPolicy<cv::cuda::GpuMat, mo::ContinuousPolicy>> allocator;
        PoolPolicy<cuda::CUDA> gpu_allocator;
        mo::cuda::AllocatorProxy<> gpu_allocator_proxy(&gpu_allocator);

        cv::cuda::GpuMat::setDefaultAllocator(&gpu_allocator_proxy);
        mo::CvAllocator<PoolPolicy<cuda::CUDA>> cpu_allocator;
        cv::Mat::setDefaultAllocator(&cpu_allocator);
        cv::cuda::Stream stream;
        start = mo::Time::now();
        mo::ScopedProfile profile("pool allocation");
        for (int i = 0; i < sizes.size(); ++i)
        {
            cv::Mat h_data(10, sizes[i], CV_32F);
            cv::cuda::GpuMat d_data;
            cv::cuda::createContinuous(10, sizes[i], CV_32F, d_data);
            d_data.upload(h_data, stream);
            cv::cuda::multiply(d_data, cv::Scalar(100), d_data, 1, -1, stream);
            cv::cuda::add(d_data, cv::Scalar(100), d_data, cv::noArray(), -1, stream);
            cv::Mat result;
            d_data.download(result, stream);
        }
        end = mo::Time::now();
        pooled_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Pooled Allocation:  " << pooled_time << "\n";
    }
}
#endif
