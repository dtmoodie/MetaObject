#include "params.hpp"
#include <RuntimeObjectSystem/ObjectInterfacePerModule.h>
namespace mo
{
    namespace params
    {
        void init(SystemTable* table)
        {
            PerModuleInterface::GetInstance()->SetSystemTable(table);
        }
    } // namespace params
} // namespace mo
