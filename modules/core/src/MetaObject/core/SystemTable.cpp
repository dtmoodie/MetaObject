

#include "SystemTable.hpp"
#include "MetaObject/logging/logging.hpp"

#include <MetaObject/thread/FiberScheduler.hpp>
#include <MetaObject/thread/ThreadPool.hpp>

#include "MetaObject/core/detail/allocator_policies/Combined.hpp"
#include "MetaObject/core/detail/allocator_policies/Lock.hpp"
#include "MetaObject/core/detail/allocator_policies/Pool.hpp"
#include "MetaObject/core/detail/allocator_policies/RefCount.hpp"
#include "MetaObject/core/detail/allocator_policies/Stack.hpp"
#include "MetaObject/core/detail/allocator_policies/Usage.hpp"

#include <MetaObject/thread/Thread.hpp>

#include <boost/fiber/condition_variable.hpp>
#include <boost/fiber/recursive_timed_mutex.hpp>

#ifdef HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

using Allocator_t =
    mo::CombinedPolicy<mo::LockPolicy<mo::PoolPolicy<mo::CPU>>, mo::LockPolicy<mo::StackPolicy<mo::CPU>>>;

std::shared_ptr<SystemTable> SystemTable::instanceImpl()
{
    static std::weak_ptr<SystemTable> inst;
    std::shared_ptr<SystemTable> output = inst.lock();
    if (!output)
    {
        output.reset(new SystemTable());
        auto logger = output->getDefaultLogger();
        logger->debug("Created new system table instance at {}", static_cast<void*>(output.get()));
        inst = output;
        initThread(*output);
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
    if (m_logger)
    {
        m_logger->debug("System table cleanup");
    }
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

void SystemTable::setDefaultAllocator(mo::AllocatorPtr_t alloc)
{
    m_default_allocator = std::move(alloc);
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

mo::IObjectTable::IObjectContainer* SystemTable::getObjectContainer(const mo::TypeInfo type) const
{
    auto itr = m_singletons.find(type);
    if (itr != m_singletons.end())
    {
        return itr->second.get();
    }
    return nullptr;
}

void SystemTable::setObjectContainer(mo::TypeInfo type, IObjectContainer::Ptr_t&& container)
{
    m_singletons[type] = std::move(container);
    MO_LOG(trace,
           "Creating new singleton instance of type {} in system table {}",
           mo::TypeTable::instance(this)->typeToName(type),
           static_cast<const void*>(this));
}

const std::shared_ptr<spdlog::logger>& SystemTable::getDefaultLogger()
{
    if (!m_logger)
    {
        auto logger = getLoggerRegistry(*this).get("default");
        if (!logger)
        {
            logger = getLoggerRegistry(*this).create("default", spdlog::sinks::stdout_sink_mt::instance());
        }
        m_logger = logger;
    }
    return m_logger;
}

void SystemTable::setDefaultLogger(const std::shared_ptr<spdlog::logger>& logger)
{
    m_logger = logger;
}
