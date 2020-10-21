#include "core.hpp"
#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"
namespace mo
{
    void initCoreModule(SystemTable* table)
    {
        PerModuleInterface::GetInstance()->SetSystemTable(table);
    }
} // namespace mo
