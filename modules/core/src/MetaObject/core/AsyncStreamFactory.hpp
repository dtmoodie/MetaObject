#ifndef MO_CORE_ASYNC_STREAM_FACTORY_HPP
#define MO_CORE_ASYNC_STREAM_FACTORY_HPP

#include <MetaObject/core/metaobject_config.hpp>
#include <MetaObject/thread/PriorityLevels.hpp>

#include <cstdint>
#include <memory>
struct SystemTable;

namespace mo
{
    struct IAsyncStream;
    struct AsyncStreamConstructor;
    struct MO_EXPORTS AsyncStreamFactory
    {
        using Ptr_t = std::shared_ptr<IAsyncStream>;

        static std::shared_ptr<AsyncStreamFactory> instance();
        static std::shared_ptr<AsyncStreamFactory> instance(SystemTable* table);

        void registerConstructor(AsyncStreamConstructor* ctr);

        Ptr_t create(const std::string& name = "",
                     int32_t device_id = 0,
                     PriorityLevels device_priority = MEDIUM,
                     PriorityLevels thread_priority = MEDIUM);

        void onStreamDestroy(const IAsyncStream* stream);

      private:
        std::vector<IAsyncStream*> m_streams;
        std::vector<AsyncStreamConstructor*> m_ctrs;
    }; // struct mo::AsyncStreamFactory
}
#endif // MO_CORE_ASYNC_STREAM_FACTORY_HPP

