#include "AsyncStreamFactory.hpp"
#include "AsyncStreamConstructor.hpp"

#include <MetaObject/core/SystemTable.hpp>
namespace mo
{
    std::shared_ptr<AsyncStreamFactory> AsyncStreamFactory::instance(SystemTable* table)
    {
        return table->getSingleton<AsyncStreamFactory>();
    }

    std::shared_ptr<AsyncStreamFactory> AsyncStreamFactory::instance()
    {
        return singleton<AsyncStreamFactory>();
    }

    void AsyncStreamFactory::registerConstructor(AsyncStreamConstructor* ctr)
    {
        m_ctrs.push_back(ctr);
    }

    AsyncStreamFactory::Ptr_t AsyncStreamFactory::create(const std::string& name,
                                                         const int32_t device_id,
                                                         const PriorityLevels device_priority,
                                                         const PriorityLevels thread_priority)
    {
        AsyncStreamConstructor* best_ctr = nullptr;
        uint32_t highest_priority = 0;
        for (auto ctr : m_ctrs)
        {
            auto p = ctr->priority(device_id);
            if (p > highest_priority)
            {
                highest_priority = p;
                best_ctr = ctr;
            }
        }
        if (best_ctr)
        {
            return best_ctr->create(name, device_id, device_priority, thread_priority);
        }
        return {};
    }
}
