#include "BufferFactory.hpp"
#include <MetaObject/core/SystemTable.hpp>
#include <MetaObject/params/InputParam.hpp>
#include <RuntimeObjectSystem/ObjectInterfacePerModule.h>
#include <map>

namespace mo
{
    namespace buffer
    {

        using CtrTable = std::map<BufferFlags, BufferFactory::BufferConstructor>;

        void BufferFactory::registerConstructor(const BufferConstructor& constructor, BufferFlags buffer)
        {
            auto instance = PerModuleInterface::GetInstance();
            if (instance)
            {
                auto table = instance->GetSystemTable();
                std::function<void(SystemTable*)> func = [constructor, buffer](SystemTable* table) {
                    CtrTable* ctr_table = singleton<CtrTable>(table);
                    (*ctr_table)[buffer] = constructor;
                };

                if (table)
                {
                    func(table);
                }
                else
                {
                    instance->AddDelayInitFunction(func);
                }
            }
        }

        InputParam* BufferFactory::createBuffer(IParam* param, BufferFlags flags)
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
    }
}
