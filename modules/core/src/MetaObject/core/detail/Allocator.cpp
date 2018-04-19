#include "MetaObject/core/detail/AllocatorImpl.hpp"
#include "MetaObject/thread/cuda.hpp"
#include <MetaObject/core/SystemTable.hpp>
#include <RuntimeObjectSystem/ObjectInterfacePerModule.h>

#include "allocator_policies/Combined-inl.hpp"
#include "allocator_policies/Lock-inl.hpp"
#include "allocator_policies/Pool-inl.hpp"
#include "allocator_policies/Stack-inl.hpp"
#include "allocator_policies/Pitched.hpp"
#include "allocator_policies/Usage-inl.hpp"
#include "allocator_policies/Continuous.hpp"
#include "allocator_policies/RefCount-inl.hpp"
#include "allocator_policies/MergedAllocator-inl.hpp"
#if MO_HAVE_OPENCV
#include "opencv_allocator-inl.hpp"
#endif
#include "singleton.hpp"
#include <ctime>

namespace mo
{
    template<class XPU>
    using Combined = UsagePolicy<RefCountPolicy<CombinedPolicy<PoolPolicy<XPU>, StackPolicy<XPU>>>>;

    template<class XPU>
    using LockedCombined = LockPolicy<Combined<XPU>>;


    unsigned char* alignMemory(unsigned char* ptr, int elem_size)
    {
        return ptr + alignmentOffset(ptr, elem_size);
    }

    const unsigned char* alignMemory(const unsigned char* ptr, int elem_size)
    {
        return ptr + alignmentOffset(ptr, elem_size);
    }

    size_t alignmentOffset(const unsigned char* ptr, size_t elem_size)
    {
        return elem_size - (reinterpret_cast<const size_t>(ptr) % elem_size);
    }

    thread_local std::string g_current_scope;

    const std::string& getScopeName() { return g_current_scope; }

    void setScopeName(const std::string& name) { g_current_scope = name; }
#if MO_HAVE_OPENCV
    typedef MergedAllocator<CvAllocator<CPU, LockedCombined<CPU>, PitchedPolicy>, CvAllocator<GPU, LockedCombined<GPU>, PitchedPolicy>> Allocator_t;
#endif

    std::shared_ptr<Allocator> Allocator::createDefaultOpencvAllocator()
    {
        std::shared_ptr<Allocator_t> allocator;
        allocator = std::make_shared<Allocator_t>();
#if MO_HAVE_OPENCV
        cv::Mat::setDefaultAllocator(allocator.get());
#if MO_OPENCV_HAVE_CUDA
        cv::cuda::GpuMat::setDefaultAllocator(allocator.get());
#endif
#endif
        return allocator;
    }

    std::shared_ptr<Allocator> Allocator::createAllocator()
    {
        std::shared_ptr<Allocator> allocator;
        allocator = std::make_shared<Allocator_t>();
        return allocator;
    }

    std::weak_ptr<Allocator> Allocator::default_allocator;

    void Allocator::setDefaultAllocator(const std::shared_ptr<Allocator>& allocator)
    {
        default_allocator = allocator;
    }

    std::shared_ptr<Allocator> Allocator::getDefaultAllocator()
    {
        return default_allocator.lock();
    }

    
} // namespace mo
