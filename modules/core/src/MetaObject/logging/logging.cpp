#include "MetaObject/logging/logging.hpp"

#include <MetaObject/core/SystemTable.hpp>
#include <RuntimeObjectSystem/ObjectInterfacePerModule.h>

void mo::initLogging()
{
}

spdlog::details::registry& getRegistry()
{
    auto table = PerModuleInterface::GetInstance()->GetSystemTable();
    if (table == nullptr)
    {
        throw std::runtime_error("SystemTable == nullptr");
    }

    return *table->getSingleton<spdlog::details::registry>();
}

spdlog::logger& mo::getDefaultLogger()
{
    return *(getRegistry().default_logger());
}
