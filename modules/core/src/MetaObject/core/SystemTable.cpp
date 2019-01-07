#include "SystemTable.hpp"
#include "MetaObject/logging/logging.hpp"
#include "singletons.hpp"

#include <MetaObject/thread/FiberScheduler.hpp>
#include <MetaObject/thread/ThreadPool.hpp>

#include "MetaObject/core/detail/allocator_policies/Combined.hpp"
#include "MetaObject/core/detail/allocator_policies/Lock.hpp"
#include "MetaObject/core/detail/allocator_policies/Pool.hpp"
#include "MetaObject/core/detail/allocator_policies/RefCount.hpp"
#include "MetaObject/core/detail/allocator_policies/Stack.hpp"
#include "MetaObject/core/detail/allocator_policies/Usage.hpp"

#ifdef HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

using Allocator_t =
    mo::CombinedPolicy<mo::LockPolicy<mo::PoolPolicy<mo::CPU>>, mo::LockPolicy<mo::StackPolicy<mo::CPU>>>;

mo::ISingletonContainer::~ISingletonContainer()
{
}

std::shared_ptr<SystemTable> SystemTable::instanceImpl()
{
    static std::weak_ptr<SystemTable> inst;
    std::shared_ptr<SystemTable> output = inst.lock();
    if (!output)
    {
        output.reset(new SystemTable());
        inst = output;
        std::shared_ptr<mo::ThreadPool> pool = mo::sharedSingleton<mo::ThreadPool>(output.get());
        boost::fibers::use_scheduling_algorithm<mo::PriorityScheduler>(pool);
    }
    return output;
}

SystemTable::SystemTable()
{
    setAllocatorConstructor([]() -> mo::AllocatorPtr_t { return std::make_shared<Allocator_t>(); });

#ifdef HAVE_CUDA
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0 || std::getenv("AQUILA_CPU_ONLY"))
    {
        m_system_info.have_cuda = false;
    }
    else
    {
        m_system_info.have_cuda = true;
    }
#endif
    auto module = PerModuleInterface::GetInstance();
    module->SetSystemTable(this);
}

SystemTable::~SystemTable()
{
    std::cout << "System table cleanup" << std::endl;
}

void SystemTable::deleteSingleton(mo::TypeInfo type)
{
    m_singletons.erase(type);
}

mo::AllocatorPtr_t SystemTable::getDefaultAllocator()
{
    if (m_default_allocator == nullptr)
    {
        m_default_allocator = createAllocator();
    }
    return m_default_allocator;
}

void SystemTable::setAllocatorConstructor(std::function<mo::AllocatorPtr_t()>&& ctr)
{
    MO_ASSERT_FMT(ctr, "Can't set an empty function for allocator construction");
    m_allocator_constructor = ctr;
}

mo::AllocatorPtr_t SystemTable::createAllocator() const
{
    return m_allocator_constructor();
}
