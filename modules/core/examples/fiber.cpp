#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"

#include <MetaObject/core/IAsyncStream.hpp>
#include <MetaObject/core/SystemTable.hpp>

#include <MetaObject/thread/FiberScheduler.hpp>
#include <MetaObject/thread/ThreadPool.hpp>

int main()
{

    auto table = SystemTable::instance();
    auto module = PerModuleInterface::GetInstance();
    module->SetSystemTable(table.get());
    std::shared_ptr<mo::ThreadPool> pool = mo::sharedSingleton<mo::ThreadPool>(table.get());
    boost::fibers::use_scheduling_algorithm<mo::PriorityScheduler>(pool);

    for (uint32_t i = 0; i < 1000; ++i)
    {
        boost::fibers::fiber f([]() {

        });
        f.detach();
    }

    boost::this_fiber::yield();
}
