#include "BufferFactory.hpp"
#include "IBuffer.hpp"
#include <MetaObject/core/SystemTable.hpp>
#include <MetaObject/params/InputParam.hpp>
#include <RuntimeObjectSystem/ObjectInterfacePerModule.h>
#include <map>

namespace mo
{
    namespace buffer
    {
        struct BufferFactoryImpl: public BufferFactory
        {
            IBuffer* createBuffer(IParam* param, BufferFlags flags)
            {
                auto ctr = ctr_table.find(flags);
                if (ctr != ctr_table.end())
                {
                    auto buffer = ctr->second();
                    if (buffer)
                    {
                        if (buffer->setInput(param))
                        {
                            return buffer;
                        }

                        delete buffer;
                    }
                }
                return nullptr;
            }

            IBuffer* createBuffer(const std::shared_ptr<IParam>& param, mo::BufferFlags buffer_type_)
            {
                auto itr = ctr_table.find(buffer_type_);
                if (itr == ctr_table.end())
                {
                    return nullptr;
                }
                auto buffer = itr->second();
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

    }
}
