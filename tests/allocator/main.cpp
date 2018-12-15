#define BOOST_TEST_MAIN
#include <MetaObject/core/detail/Allocator.hpp>
#include <MetaObject/core/detail/StlAllocator.hpp>
#include <MetaObject/core/detail/Time.hpp>

#include <MetaObject/core/detail/allocator_policies/Combined.hpp>
#include <MetaObject/core/detail/allocator_policies/Continuous.hpp>
#include <MetaObject/core/detail/allocator_policies/Lock.hpp>
#include <MetaObject/core/detail/allocator_policies/Pool.hpp>
#include <MetaObject/core/detail/allocator_policies/Stack.hpp>
#include <MetaObject/core/detail/allocator_policies/opencv.hpp>

#include <MetaObject/cuda/CvAllocator.hpp>
#include <MetaObject/cuda/MemoryBlock.hpp>

#include <MetaObject/logging/logging.hpp>
#include <MetaObject/logging/profiling.hpp>

#include <opencv2/cudaarithm.hpp>

#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE "MetaObjectInheritance"
#include <boost/test/included/unit_test.hpp>
#endif

#include <iostream>

using namespace mo;

BOOST_AUTO_TEST_CASE(initialize_thread)
{
    // TODO fix unit test
    // BOOST_REQUIRE(mo::Allocator::getDefaultAllocator());
    mo::initLogging();
    mo::initProfiling();
}

BOOST_AUTO_TEST_CASE(initialize_global)
{
    // TODO fix unit test
    // BOOST_REQUIRE(mo::Allocator::getDefaultAllocator());
}

BOOST_AUTO_TEST_CASE(test_cpu_set_allocator)
{
}

BOOST_AUTO_TEST_CASE(test_gpu_set_allocator)
{
}

BOOST_AUTO_TEST_CASE(test_cpu_pooled_allocation)
{
    auto start = mo::Time::now();

    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(1, 100 * std::min(1000, 1 + rand()), CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    auto end = mo::Time::now();
    auto non_pinned_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    start = mo::Time::now();
    mo::CvAllocator<mo::cuda::HOST> pinned_allocator;
    cv::Mat::setDefaultAllocator(&pinned_allocator);
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(1, 100 * std::min(1000, 1 + rand()), CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = mo::Time::now();
    auto non_pooled_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    mo::CvAllocator<mo::PoolPolicy<mo::CPU>> pooled_allocator;
    cv::Mat::setDefaultAllocator(&pooled_allocator);

    start = mo::Time::now();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(1, 100 * std::min(1000, 1 + rand()), CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = mo::Time::now();
    auto pooled_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Test the thread safe allocators
    mo::CvAllocator<mo::LockPolicy<mo::PoolPolicy<mo::CPU>>> mt_pooled_allocator;
    cv::Mat::setDefaultAllocator(&mt_pooled_allocator);

    start = mo::Time::now();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(1, 100 * std::min(1000, 1 + rand()), CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = mo::Time::now();
    auto mt_pooled_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cv::Mat::setDefaultAllocator(nullptr);

    BOOST_REQUIRE_LT(pooled_time, non_pooled_time);
    std::cout << "\n ======================================================================== \n";
    std::cout << " Random Allocation Pattern\n";
    std::cout << " Default Allocator Time: " << non_pinned_time << "\n"
              << " Pinned Allocator Time:  " << non_pooled_time << "\n"
              << " Pooled Time:            " << pooled_time << "\n"
              << " Thead Safe Pooled Time: " << mt_pooled_time;
}

BOOST_AUTO_TEST_CASE(test_cpu_stack_allocation)
{
    auto start = mo::Time::now();
    cv::Mat zeroAlloc(2000, 2000, CV_32FC2);
    for (int i = 0; i < 1000; ++i)
    {
        zeroAlloc *= 100;
        zeroAlloc += 10;
    }
    auto end = mo::Time::now();
    const auto zero_alloc_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    start = mo::Time::now();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2000, 2000, CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = mo::Time::now();
    const auto non_pinned_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    start = mo::Time::now();
    mo::CvAllocator<mo::cuda::HOST> pinned_allocator;
    cv::Mat::setDefaultAllocator(&pinned_allocator);
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2000, 2000, CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = mo::Time::now();
    const auto non_pooled_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    mo::CvAllocator<mo::StackPolicy<mo::cuda::HOST>> allocator;
    cv::Mat::setDefaultAllocator(&allocator);

    start = mo::Time::now();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2000, 2000, CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = mo::Time::now();
    const auto pooled_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Test the thread safe allocators
    mo::CvAllocator<mo::LockPolicy<mo::StackPolicy<mo::cuda::HOST>>> mt_stack_allocator;
    cv::Mat::setDefaultAllocator(&mt_stack_allocator);

    start = mo::Time::now();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2000, 2000, CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = mo::Time::now();
    const auto mt_pooled_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cv::Mat::setDefaultAllocator(nullptr);

    BOOST_REQUIRE_LT(pooled_time, non_pooled_time);
    std::cout << "\n ======================================================================== \n";
    std::cout << " Fixed Allocation Pattern\n";
    std::cout << " Default Allocator Time:   " << non_pinned_time << "\n"
              << " Pinned Allocator Time:    " << non_pooled_time << "\n"
              << " Pooled Time:              " << pooled_time << "\n"
              << " Thead Safe Pooled Time:   " << mt_pooled_time << "\n"
              << " Zero Allocation Time:     " << zero_alloc_time;
}

BOOST_AUTO_TEST_CASE(test_cpu_combined_allocation)
{
    mo::CvAllocator<CombinedPolicy<PoolPolicy<CPU>, StackPolicy<CPU>>> allocator;
    cv::Mat::setDefaultAllocator(&allocator);
    auto start = mo::Time::now();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2, std::min(1000, 1 + rand()) * 100, CV_32F);
        vec *= 100;
        vec += 10;
    }
    auto end = mo::Time::now();
    double random_size = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    start = mo::Time::now();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2000, 2000, CV_32F);
        vec *= 100;
        vec += 10;
    }
    end = mo::Time::now();
    const auto set_size = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    mo::CvAllocator<LockPolicy<CombinedPolicy<PoolPolicy<CPU>, StackPolicy<CPU>>>> mt_allocator;
    cv::Mat::setDefaultAllocator(&allocator);
    start = mo::Time::now();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2, std::min(1000, 1 + rand()) * 100, CV_32F);
        vec *= 100;
        vec += 10;
    }
    end = mo::Time::now();
    const double mt_random_size = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    start = mo::Time::now();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2000, 2000, CV_32F);
        vec *= 100;
        vec += 10;
    }
    end = mo::Time::now();
    const double mt_set_size = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "\n ======================================================================== \n";
    std::cout << "------------ Thread specifc ---------------\n"
              << " Random Allocation Pattern: " << random_size << "\n"
              << " Set Allocation Pattern:    " << set_size << "\n"
              << "------------ Thread safe ---------------\n"
              << " Random Allocation Pattern: " << mt_random_size << "\n"
              << " Set Allocation Pattern:    " << mt_set_size << "\n";
}

BOOST_AUTO_TEST_CASE(test_gpu_random_allocation_pattern)
{
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
    double default_allocator = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    start = mo::Time::now();

    mo::cuda::CvAllocator<CombinedPolicy<PoolPolicy<cuda::CUDA>, StackPolicy<cuda::CUDA>>> combined_allocator;
    cv::cuda::GpuMat::setDefaultAllocator(&combined_allocator);
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
              << " Default Allocator: " << default_allocator << "\n"
              << " Pool Allocator:    " << pool_allocator << "\n";

    // cv::cuda::GpuMat::setDefaultAllocator(defaultAllocator);
    // Allocator::getThreadSpecificAllocator()->Release();
}

BOOST_AUTO_TEST_CASE(test_gpu_static_allocation_pattern)
{
    cv::cuda::Stream stream;

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
    mo::cuda::CvAllocator<PoolPolicy<cuda::CUDA>> pool_allocator;
    auto defaultAllocator = cv::cuda::GpuMat::defaultAllocator();
    cv::cuda::GpuMat::setDefaultAllocator(&pool_allocator);
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

    cv::cuda::GpuMat::setDefaultAllocator(defaultAllocator);
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

BOOST_AUTO_TEST_CASE(async_transfer_rate_random)
{
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
        mo::scoped_profile profile("zero allocation");
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
        mo::scoped_profile profile("default allocation");
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
        mo::cuda::CvAllocator<PoolPolicy<cuda::CUDA>> gpu_allocator;
        // auto defaultAllocator = cv::cuda::GpuMat::defaultAllocator();
        cv::cuda::GpuMat::setDefaultAllocator(&gpu_allocator);
        mo::CvAllocator<PoolPolicy<cuda::CUDA>> cpu_allocator;
        cv::Mat::setDefaultAllocator(&cpu_allocator);
        cv::cuda::Stream stream;
        start = mo::Time::now();
        mo::scoped_profile profile("pool allocation");
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
