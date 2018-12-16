#include <MetaObject/core/detail/Allocator.hpp>
#include <MetaObject/core/detail/StlAllocator.hpp>
#include <MetaObject/core/detail/Time.hpp>

#include <MetaObject/core/detail/allocator_policies/Combined.hpp>
#include <MetaObject/core/detail/allocator_policies/Continuous.hpp>
#include <MetaObject/core/detail/allocator_policies/Lock.hpp>
#include <MetaObject/core/detail/allocator_policies/Pool.hpp>
#include <MetaObject/core/detail/allocator_policies/Stack.hpp>
#include <MetaObject/core/detail/allocator_policies/opencv.hpp>

#include <boost/test/auto_unit_test.hpp>
#include <boost/test/test_tools.hpp>

using namespace mo;

BOOST_AUTO_TEST_CASE(test_cpu_pooled_allocation)
{
    auto start = Time::now();

    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(1, 100 * std::min(1000, 1 + rand()), CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    auto end = Time::now();
    auto non_pinned_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    start = Time::now();
    CvAllocator<CPU> pinned_allocator;
    cv::Mat::setDefaultAllocator(&pinned_allocator);
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(1, 100 * std::min(1000, 1 + rand()), CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = Time::now();
    auto non_pooled_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    CvAllocator<PoolPolicy<CPU>> pooled_allocator;
    cv::Mat::setDefaultAllocator(&pooled_allocator);

    start = Time::now();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(1, 100 * std::min(1000, 1 + rand()), CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = Time::now();
    auto pooled_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Test the thread safe allocators
    CvAllocator<LockPolicy<PoolPolicy<CPU>>> mt_pooled_allocator;
    cv::Mat::setDefaultAllocator(&mt_pooled_allocator);

    start = Time::now();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(1, 100 * std::min(1000, 1 + rand()), CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = Time::now();
    auto mt_pooled_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cv::Mat::setDefaultAllocator(nullptr);

    // BOOST_REQUIRE_LT(pooled_time, non_pooled_time);
    std::cout << "\n ======================================================================== \n";
    std::cout << " Random Allocation Pattern\n";
    std::cout << " Default Allocator Time: " << non_pinned_time << "\n"
              << " Pinned Allocator Time:  " << non_pooled_time << "\n"
              << " Pooled Time:            " << pooled_time << "\n"
              << " Thead Safe Pooled Time: " << mt_pooled_time;
}

BOOST_AUTO_TEST_CASE(test_cpu_stack_allocation)
{
    auto start = Time::now();
    cv::Mat zeroAlloc(2000, 2000, CV_32FC2);
    for (int i = 0; i < 1000; ++i)
    {
        zeroAlloc *= 100;
        zeroAlloc += 10;
    }
    auto end = Time::now();
    const auto zero_alloc_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    start = Time::now();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2000, 2000, CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = Time::now();
    const auto non_pinned_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    start = Time::now();
    CvAllocator<CPU> pinned_allocator;
    cv::Mat::setDefaultAllocator(&pinned_allocator);
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2000, 2000, CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = Time::now();
    const auto non_pooled_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    CvAllocator<StackPolicy<CPU>> allocator;
    cv::Mat::setDefaultAllocator(&allocator);

    start = Time::now();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2000, 2000, CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = Time::now();
    const auto pooled_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Test the thread safe allocators
    CvAllocator<LockPolicy<StackPolicy<CPU>>> mt_stack_allocator;
    cv::Mat::setDefaultAllocator(&mt_stack_allocator);

    start = Time::now();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2000, 2000, CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = Time::now();
    const auto mt_pooled_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cv::Mat::setDefaultAllocator(nullptr);

    // BOOST_REQUIRE_LT(pooled_time, non_pooled_time);
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
    CvAllocator<CombinedPolicy<PoolPolicy<CPU>, StackPolicy<CPU>>> allocator;
    cv::Mat::setDefaultAllocator(&allocator);
    auto start = Time::now();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2, std::min(1000, 1 + rand()) * 100, CV_32F);
        vec *= 100;
        vec += 10;
    }
    auto end = Time::now();
    double random_size = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    start = Time::now();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2000, 2000, CV_32F);
        vec *= 100;
        vec += 10;
    }
    end = Time::now();
    const auto set_size = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    CvAllocator<LockPolicy<CombinedPolicy<PoolPolicy<CPU>, StackPolicy<CPU>>>> mt_allocator;
    cv::Mat::setDefaultAllocator(&allocator);
    start = Time::now();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2, std::min(1000, 1 + rand()) * 100, CV_32F);
        vec *= 100;
        vec += 10;
    }
    end = Time::now();
    const double mt_random_size = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    start = Time::now();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2000, 2000, CV_32F);
        vec *= 100;
        vec += 10;
    }
    end = Time::now();
    const double mt_set_size = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "\n ======================================================================== \n";
    std::cout << "------------ Thread specifc ---------------\n"
              << " Random Allocation Pattern: " << random_size << "\n"
              << " Set Allocation Pattern:    " << set_size << "\n"
              << "------------ Thread safe ---------------\n"
              << " Random Allocation Pattern: " << mt_random_size << "\n"
              << " Set Allocation Pattern:    " << mt_set_size << "\n";
}
