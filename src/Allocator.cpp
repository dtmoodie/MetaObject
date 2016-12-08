#include "MetaObject/Detail/AllocatorImpl.hpp"

using namespace mo;
boost::thread_specific_ptr<Allocator> thread_specific_allocator;

//typedef ConcreteAllocator<LockPolicy<PoolPolicy<PitchedPolicy>>> LockedPitchedPool_t;
//typedef ConcreteAllocator<LockPolicy<PoolPolicy<ContinuousPolicy>>> LockedPitchedPool_t;

//typedef ConcreteAllocator<LockPolicy<StackPolicy<ContinuousPolicy>>> LockedPitchedStack_t;
//typedef ConcreteAllocator<LockPolicy<StackPolicy<PitchedPolicy>>> LockedPitchedStack_t;

//typedef ConcreteAllocator<StackPolicy<ContinuousPolicy>> PitchedStack_t;
//typedef ConcreteAllocator<StackPolicy<PitchedPolicy>> PitchedStack_t;




Allocator* Allocator::GetThreadSafeAllocator()
{
    return nullptr;
}

Allocator* Allocator::GetThreadSpecificAllocator()
{
    if(thread_specific_allocator.get() == nullptr)
    {
        thread_specific_allocator.reset();
    }
    thread_specific_allocator.get();
}
