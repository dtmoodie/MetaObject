#include "core.hpp"
#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"
namespace mo
{
    void initCoreModule(SystemTable* table)
    {
        PerModuleInterface* instance = PerModuleInterface::GetInstance();
        instance->SetSystemTable(table);
    }
} // namespace mo
