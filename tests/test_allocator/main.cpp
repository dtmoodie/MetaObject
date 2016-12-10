#define BOOST_TEST_MAIN
#include "MetaObject/Detail/Allocator.hpp"
#include "MetaObject/Detail/AllocatorImpl.hpp"
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/common.hpp>
#include <boost/log/exceptions.hpp>
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
    boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::warning);
    BOOST_REQUIRE(mo::Allocator::GetThreadSpecificAllocator());
}

BOOST_AUTO_TEST_CASE(initialize_global)
{
    BOOST_REQUIRE(mo::Allocator::GetThreadSafeAllocator());
}



BOOST_AUTO_TEST_CASE(test_cpu_pooled_allocation)
{
    auto start = boost::posix_time::microsec_clock::local_time();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(1, 100 * std::min(1000, 1 + rand()), CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    auto end = boost::posix_time::microsec_clock::local_time();
    double non_pinned_time = boost::posix_time::time_duration(end - start).total_milliseconds();

    start = boost::posix_time::microsec_clock::local_time();
    mo::PinnedAllocator pinnedAllocator;
    cv::Mat::setDefaultAllocator(&pinnedAllocator);
    for(int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(1, 100 * std::min(1000, 1 + rand()), CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = boost::posix_time::microsec_clock::local_time();
    double non_pooled_time = boost::posix_time::time_duration(end - start).total_milliseconds();
    mo::CpuPoolPolicy allocator;
    cv::Mat::setDefaultAllocator(&allocator);

    start = boost::posix_time::microsec_clock::local_time();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(1, 100 * std::min(1000, 1 + rand()), CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = boost::posix_time::microsec_clock::local_time();
    double pooled_time = boost::posix_time::time_duration(end - start).total_milliseconds();
    
    

    // Test the thread safe allocators
    mo::mt_CpuPoolPolicy mtPoolAllocator;
    cv::Mat::setDefaultAllocator(&mtPoolAllocator);

    start = boost::posix_time::microsec_clock::local_time();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(1, 100 * std::min(1000, 1 + rand()), CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = boost::posix_time::microsec_clock::local_time();
    double mt_pooled_time = boost::posix_time::time_duration(end - start).total_milliseconds();
    cv::Mat::setDefaultAllocator(nullptr);

    BOOST_REQUIRE_LT(pooled_time, non_pooled_time);
    std::cout << "\n ======================================================================== \n";
    std::cout << " Random Allocation Pattern\n";
    std::cout 
        << " Default Allocator Time: " << non_pinned_time << "\n"
        << " Pinned Allocator Time:  " << non_pooled_time << "\n"
        << " Pooled Time:            " << pooled_time << "\n"
        << " Thead Safe Pooled Time: " << mt_pooled_time;
}

BOOST_AUTO_TEST_CASE(test_cpu_stack_allocation)
{
    auto start = boost::posix_time::microsec_clock::local_time();
    cv::Mat zeroAlloc(2000, 2000, CV_32FC2);
    for (int i = 0; i < 1000; ++i)
    {
        zeroAlloc *= 100;
        zeroAlloc += 10;
    }
    auto end = boost::posix_time::microsec_clock::local_time();
    double zero_alloc_time = boost::posix_time::time_duration(end - start).total_milliseconds();

    start = boost::posix_time::microsec_clock::local_time();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2000, 2000, CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = boost::posix_time::microsec_clock::local_time();
    double non_pinned_time = boost::posix_time::time_duration(end - start).total_milliseconds();

    start = boost::posix_time::microsec_clock::local_time();
    mo::PinnedAllocator pinnedAllocator;
    cv::Mat::setDefaultAllocator(&pinnedAllocator);
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2000, 2000, CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = boost::posix_time::microsec_clock::local_time();
    double non_pooled_time = boost::posix_time::time_duration(end - start).total_milliseconds();
    mo::CpuStackPolicy allocator;
    cv::Mat::setDefaultAllocator(&allocator);

    start = boost::posix_time::microsec_clock::local_time();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2000, 2000, CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = boost::posix_time::microsec_clock::local_time();
    double pooled_time = boost::posix_time::time_duration(end - start).total_milliseconds();

    // Test the thread safe allocators
    mo::mt_CpuStackPolicy mtPoolAllocator;
    cv::Mat::setDefaultAllocator(&mtPoolAllocator);

    start = boost::posix_time::microsec_clock::local_time();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2000, 2000, CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = boost::posix_time::microsec_clock::local_time();
    double mt_pooled_time = boost::posix_time::time_duration(end - start).total_milliseconds();
    cv::Mat::setDefaultAllocator(nullptr);

    BOOST_REQUIRE_LT(pooled_time, non_pooled_time);
    std::cout << "\n ======================================================================== \n";
    std::cout << " Fixed Allocation Pattern\n";
    std::cout 
        << " Default Allocator Time:   " << non_pinned_time << "\n"
        << " Pinned Allocator Time:    " << non_pooled_time << "\n"
        << " Pooled Time:              " << pooled_time << "\n"
        << " Thead Safe Pooled Time:   " << mt_pooled_time << "\n"
        << " Zero Allocation Time:     " << zero_alloc_time;
}
BOOST_AUTO_TEST_CASE(test_cpu_combined_allocation)
{
    cv::Mat::setDefaultAllocator(mo::Allocator::GetThreadSpecificAllocator());
    auto start = boost::posix_time::microsec_clock::local_time();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2, std::min(1000, 1 + rand()) * 100, CV_32F);
        vec *= 100;
        vec += 10;
    }
    auto end = boost::posix_time::microsec_clock::local_time();
    double random_size = boost::posix_time::time_duration(end - start).total_milliseconds();

    start = boost::posix_time::microsec_clock::local_time();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2000, 2000, CV_32F);
        vec *= 100;
        vec += 10;
    }
    end = boost::posix_time::microsec_clock::local_time();
    double set_size = boost::posix_time::time_duration(end - start).total_milliseconds();

    cv::Mat::setDefaultAllocator(mo::Allocator::GetThreadSafeAllocator());
    start = boost::posix_time::microsec_clock::local_time();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2, std::min(1000, 1 + rand()) * 100, CV_32F);
        vec *= 100;
        vec += 10;
    }
    end = boost::posix_time::microsec_clock::local_time();
    double mt_random_size = boost::posix_time::time_duration(end - start).total_milliseconds();

    start = boost::posix_time::microsec_clock::local_time();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2000, 2000, CV_32F);
        vec *= 100;
        vec += 10;
    }
    end = boost::posix_time::microsec_clock::local_time();
    double mt_set_size = boost::posix_time::time_duration(end - start).total_milliseconds();


    std::cout << "\n ======================================================================== \n";
    std::cout
        << "------------ Thread specifc ---------------\n"
        << " Random Allocation Pattern: " << random_size << "\n"
        << " Set Allocation Pattern:    " << set_size << "\n"
        << "------------ Thread safe ---------------\n"
        << " Random Allocation Pattern: " << mt_random_size << "\n"
        << " Set Allocation Pattern:    " << mt_set_size << "\n";
}

BOOST_AUTO_TEST_CASE(test_gpu_random_allocation_pattern)
{
    cv::cuda::Stream stream;
    auto start = boost::posix_time::microsec_clock::local_time();
    for(int i = 0; i < 1000; ++i)
    {
        cv::cuda::GpuMat X(1, std::min(1000, 1 + rand()), CV_32F);
        cv::cuda::GpuMat Y;
        cv::cuda::multiply(X, cv::Scalar(100), Y, 1, -1, stream);
        cv::cuda::subtract(Y, cv::Scalar(100), Y, cv::noArray(), -1, stream);
    }
    auto end = boost::posix_time::microsec_clock::local_time();
    double default_allocator = boost::posix_time::time_duration(end - start).total_milliseconds();

    start = boost::posix_time::microsec_clock::local_time();
    ConcreteAllocator<h_PoolAllocator_t, d_TensorPoolAllocator_t> poolAllocator;
    auto defaultAllocator = cv::cuda::GpuMat::defaultAllocator();
    cv::cuda::GpuMat::setDefaultAllocator(&poolAllocator);
    for(int i = 0; i < 1000; ++i)
    {
        cv::cuda::GpuMat X(1, std::min(1000, 1 + rand()), CV_32F);
        cv::cuda::GpuMat Y;
        cv::cuda::multiply(X, cv::Scalar(100), Y, 1, -1, stream);
        cv::cuda::subtract(Y, cv::Scalar(100), Y, cv::noArray(), -1, stream);
    }

    end = boost::posix_time::microsec_clock::local_time();
    double pool_allocator = boost::posix_time::time_duration(end - start).total_milliseconds();

    std::cout << "\n ======================= GPU ============================================ \n";
    std::cout
        << "------------ Thread specifc ---------------\n"
        << " Default Allocator: " << default_allocator<< "\n"
        << " Pool Allocator:    " << pool_allocator << "\n";

    poolAllocator.Release();
    cv::cuda::GpuMat::setDefaultAllocator(defaultAllocator);
    //Allocator::GetThreadSpecificAllocator()->Release();
}


BOOST_AUTO_TEST_CASE(test_gpu_static_allocation_pattern)
{
    boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::debug);
    cv::cuda::Stream stream;
    auto start = boost::posix_time::microsec_clock::local_time();
    for(int i = 0; i < 1000; ++i)
    {
        cv::cuda::GpuMat X(2000, 2000, CV_32F);
        cv::cuda::GpuMat Y;
        cv::cuda::multiply(X, cv::Scalar(100), Y, 1, -1, stream);
        cv::cuda::subtract(Y, cv::Scalar(100), Y, cv::noArray(), -1, stream);
    }
    auto end = boost::posix_time::microsec_clock::local_time();
    double default_allocator = boost::posix_time::time_duration(end - start).total_milliseconds();

    start = boost::posix_time::microsec_clock::local_time();
    ConcreteAllocator<h_PoolAllocator_t, d_TextureAllocator_t> poolAllocator;
    auto defaultAllocator = cv::cuda::GpuMat::defaultAllocator();
    cv::cuda::GpuMat::setDefaultAllocator(&poolAllocator);
    for(int i = 0; i < 1000; ++i)
    {
        cv::cuda::GpuMat X(2000, 2000, CV_32F);
        cv::cuda::GpuMat Y;
        cv::cuda::multiply(X, cv::Scalar(100), Y, 1, -1, stream);
        cv::cuda::subtract(Y, cv::Scalar(100), Y, cv::noArray(), -1, stream);
    }

    end = boost::posix_time::microsec_clock::local_time();
    double pool_allocator = boost::posix_time::time_duration(end - start).total_milliseconds();

    std::cout << "\n ======================= GPU ============================================ \n";
    std::cout
        << "------------ Thread specifc ---------------\n"
        << " Default Allocator: " << default_allocator<< "\n"
        << " Pool Allocator:    " << pool_allocator << "\n";

    poolAllocator.Release();
    Allocator::GetThreadSpecificAllocator()->Release();
    cv::cuda::GpuMat::setDefaultAllocator(defaultAllocator);
}










