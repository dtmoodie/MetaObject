#include "BufferFactory.hpp"
#include "IBuffer.hpp"
#include <MetaObject/core/SystemTable.hpp>
#include <MetaObject/params/ISubscriber.hpp>
#include <RuntimeObjectSystem/ObjectInterfacePerModule.h>
#include <map>

namespace mo
{
    namespace buffer
    {
        struct BufferFactoryImpl : public BufferFactory
        {
            std::shared_ptr<IBuffer> createBuffer(IPublisher& param, BufferFlags flags)
            {
                auto ctr = ctr_table.find(flags);
                if (ctr != ctr_table.end())
                {
                    std::shared_ptr<IBuffer> buffer(ctr->second());
                    if (buffer)
                    {
                        if (buffer->setInput(&param))
                        {
                            return buffer;
                        }
                    }
                }
                return nullptr;
            }

            std::shared_ptr<IBuffer> createBuffer(const std::shared_ptr<IPublisher>& param,
                                                  mo::BufferFlags buffer_type_)
            {
                auto itr = ctr_table.find(buffer_type_);
                if (itr == ctr_table.end())
                {
                    return nullptr;
                }
                std::shared_ptr<IBuffer> buffer(itr->second());
                if (buffer)
                {
                    if (buffer->setInput(param))
                    {
                        return buffer;
                    }
                }
                return nullptr;
            }

            void registerConstructorImpl(const BufferConstructor& constructor, BufferFlags buffer)
            {
                ctr_table[buffer] = constructor;
            }

          private:
            std::map<BufferFlags, BufferFactory::BufferConstructor> ctr_table;
        };

        void BufferFactory::registerConstructor(const BufferConstructor& constructor, BufferFlags buffer)
        {
            SystemTable::dispatchToSystemTable([constructor, buffer](SystemTable* table) {
                BufferFactory::instance(table)->registerConstructorImpl(constructor, buffer);
            });
        }

        std::shared_ptr<BufferFactory> BufferFactory::instance(SystemTable* table)
        {
            return table->getSingleton<BufferFactory, BufferFactoryImpl>();
        }

    } // namespace buffer
} // namespace mo
