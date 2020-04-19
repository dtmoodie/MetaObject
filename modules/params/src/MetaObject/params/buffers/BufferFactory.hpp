#pragma once
#include "IBuffer.hpp"

#include "MetaObject/core/detail/Enums.hpp"
#include "MetaObject/detail/Export.hpp"
#include <MetaObject/core/SystemTable.hpp>
#include <MetaObject/detail/defines.hpp>

#include <functional>

namespace std
{
    template <class T>
    class shared_ptr;
}

struct SystemTable;
namespace mo
{
    class IParam;
    class ISubscriber;
    class IPublisher;

    namespace buffer
    {
        class MO_EXPORTS BufferFactory
        {
          public:
            using BufferConstructor = std::function<IBuffer*()>;

            static std::shared_ptr<BufferFactory> instance(SystemTable*);
            MO_INLINE static std::shared_ptr<BufferFactory> instance();
            static void registerConstructor(const BufferConstructor& constructor, BufferFlags buffer);
            virtual std::shared_ptr<IBuffer> createBuffer(IPublisher& param, BufferFlags flags) = 0;
            virtual std::shared_ptr<IBuffer> createBuffer(const std::shared_ptr<IPublisher>& param, BufferFlags flags) = 0;

          private:
            virtual void registerConstructorImpl(const BufferConstructor& constructor, BufferFlags buffer) = 0;
        };

        std::shared_ptr<BufferFactory> BufferFactory::instance()
        {
            return singleton<BufferFactory>();
        }
    } // namespace buffer
} // namespace mo
