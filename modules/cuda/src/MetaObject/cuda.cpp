#include "cuda.hpp"
#include <MetaObject/core/SystemTable.hpp>
#include <MetaObject/cuda/MemoryBlock.hpp>

#include "MetaObject/core/detail/allocator_policies/Combined.hpp"
#include "MetaObject/core/detail/allocator_policies/Lock.hpp"
#include "MetaObject/core/detail/allocator_policies/Pool.hpp"
#include "MetaObject/core/detail/allocator_policies/RefCount.hpp"
#include "MetaObject/core/detail/allocator_policies/Stack.hpp"
#include "MetaObject/core/detail/allocator_policies/Usage.hpp"

namespace mo
{
    namespace cuda
    {
        template <class T>
        using AllocatorPolicies_t =
            mo::CombinedPolicy<mo::LockPolicy<mo::PoolPolicy<T>>, mo::LockPolicy<mo::StackPolicy<T>>>;
        void init(SystemTable* table)
        {
            table->registerModule();
            using Allocator_t = AllocatorPolicies_t<mo::cuda::HOST>;
            table->setAllocatorConstructor([]() -> mo::AllocatorPtr_t { return std::make_shared<Allocator_t>(); });

            std::shared_ptr<mo::DeviceAllocator> device_allocator =
                std::make_shared<AllocatorPolicies_t<mo::cuda::CUDA>>();
            table->setSingleton(std::move(device_allocator));
        }
    }
}

extern "C" {
void initModule(SystemTable* table)
{
    mo::cuda::init(table);
}
}
