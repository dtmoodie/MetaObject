

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
        auto logger = output->getLogger();
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
    m_singletons.clear();
    if (m_logger)
    {
        m_logger->debug("System table cleanup");
    }
}

void SystemTable::deleteSingleton(mo::TypeInfo type)
{
    std::lock_guard<std::recursive_mutex> lock(m_mtx);
    m_singletons.erase(type);
}

mo::AllocatorPtr_t SystemTable::getDefaultAllocator()
{
    std::lock_guard<std::recursive_mutex> lock(m_mtx);
    if (m_default_allocator == nullptr)
    {
        m_default_allocator = createAllocator();
    }
    return m_default_allocator;
}

void SystemTable::setDefaultAllocator(mo::AllocatorPtr_t alloc)
{
    std::lock_guard<std::recursive_mutex> lock(m_mtx);
    m_default_allocator = std::move(alloc);
}

void SystemTable::setAllocatorConstructor(std::function<mo::AllocatorPtr_t()>&& ctr)
{
    MO_ASSERT_FMT(ctr, "Can't set an empty function for allocator construction");
    std::lock_guard<std::recursive_mutex> lock(m_mtx);
    m_allocator_constructor = ctr;
}

mo::AllocatorPtr_t SystemTable::createAllocator() const
{
    std::lock_guard<std::recursive_mutex> lock(m_mtx);
    return m_allocator_constructor();
}

mo::IObjectTable::IObjectContainer* SystemTable::getObjectContainer(const mo::TypeInfo type) const
{
    std::lock_guard<std::recursive_mutex> lock(m_mtx);
    auto itr = m_singletons.find(type);
    if (itr != m_singletons.end())
    {
        return itr->second.get();
    }
    return nullptr;
}

void SystemTable::setObjectContainer(mo::TypeInfo type, IObjectContainer::Ptr_t&& container)
{
    std::lock_guard<std::recursive_mutex> lock(m_mtx);
    m_singletons[type] = std::move(container);
    MO_LOG(trace,
           "Creating new singleton instance of type {} in system table {}",
           mo::TypeTable::instance(this)->typeToName(type),
           static_cast<const void*>(this));
}

std::shared_ptr<spdlog::logger> SystemTable::getLogger(const std::string& name)
{
    std::lock_guard<std::recursive_mutex> lock(m_mtx);
    if (name == "default")
    {
        if (!m_logger)
        {
            auto logger = getLoggerRegistry(*this).get(name);
            if (!logger)
            {
                logger = getLoggerRegistry(*this).create(name, spdlog::sinks::stdout_sink_mt::instance());
            }
            m_logger = logger;
        }
        return m_logger;
    }
    else
    {
        auto logger = getLoggerRegistry(*this).get(name);
        if (!logger)
        {
            logger = getLoggerRegistry(*this).create(name, spdlog::sinks::stdout_sink_mt::instance());
        }
        return logger;
    }
}

void SystemTable::setDefaultLogger(const std::shared_ptr<spdlog::logger>& logger)
{
    std::lock_guard<std::recursive_mutex> lock(m_mtx);
    m_logger = logger;
}
