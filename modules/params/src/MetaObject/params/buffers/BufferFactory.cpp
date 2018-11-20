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
        struct CtrTable : public std::map<BufferFlags, BufferFactory::BufferConstructor>
        {
        };

        void BufferFactory::registerConstructor(const BufferConstructor& constructor, BufferFlags buffer)
        {
            SystemTable::staticDispatchToSystemTable([constructor, buffer](SystemTable* table) {
                CtrTable* ctr_table = singleton<CtrTable>(table);
                (*ctr_table)[buffer] = constructor;
            });
        }

        IBuffer* BufferFactory::createBuffer(IParam* param, BufferFlags flags)
        {
            auto instance = PerModuleInterface::GetInstance();
            if (!instance)
            {
                return nullptr;
            }
            auto table = instance->GetSystemTable();
            if (!table)
            {
                return nullptr;
            }

            auto ctr_table = singleton<CtrTable>();

            auto ctr = ctr_table->find(flags);
            if (ctr != ctr_table->end())
            {
                auto buffer = ctr->second();
                if (buffer)
                {
                    if (buffer->setInput(param))
                    {
                        return buffer;
                    }
                    else
                    {
                        delete buffer;
                    }
                }
            }
            return nullptr;
        }

        IBuffer* BufferFactory::createBuffer(const std::shared_ptr<IParam>& param, mo::BufferFlags buffer_type_)
        {
            auto instance = PerModuleInterface::GetInstance();
            if (!instance)
            {
                return nullptr;
            }
            auto table = instance->GetSystemTable();
            if (!table)
            {
                return nullptr;
            }

            auto ctr_table = singleton<CtrTable>();

            auto itr = ctr_table->find(buffer_type_);
            if (itr == ctr_table->end())
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
    }
}
