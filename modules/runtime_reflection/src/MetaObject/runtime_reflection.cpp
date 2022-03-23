#include "runtime_reflection.hpp"
#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"

namespace mo
{
    namespace runtime_reflection
    {
        void initModule(SystemTable* table)
        {
            PerModuleInterface::GetInstance()->SetSystemTable(table);
        }
    } // namespace runtime_reflection
} // namespace mo

extern "C" {
void initModuleWithSystemTable(SystemTable* table)
{
    mo::runtime_reflection::initModule(table);
}
}