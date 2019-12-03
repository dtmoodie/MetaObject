#include "MetaObject/logging/logging.hpp"

#include <MetaObject/core/SystemTable.hpp>
#include <RuntimeObjectSystem/ObjectInterfacePerModule.h>

void mo::initLogging()
{
}

namespace mo
{
    template <>
    struct ObjectConstructor<spdlog::details::registry>
    {
        std::shared_ptr<spdlog::details::registry> createShared()
        {
            return std::shared_ptr<spdlog::details::registry>(&spdlog::details::registry::instance(),
                                                              [](spdlog::details::registry*) {});
        }
    };
}

spdlog::details::registry& mo::getLoggerRegistry()
{
    auto table = PerModuleInterface::GetInstance()->GetSystemTable();
    if (table == nullptr)
    {
        throw std::runtime_error("SystemTable == nullptr");
    }

    auto registry = table->getSingleton<spdlog::details::registry>();
    return *registry;
}

spdlog::logger& mo::getDefaultLogger()
{
    auto table = PerModuleInterface::GetInstance()->GetSystemTable();
    if (table == nullptr)
    {
        throw std::runtime_error("SystemTable == nullptr");
    }
    return *table->getDefaultLogger();
}

std::shared_ptr<spdlog::logger> mo::getLogger()
{
    auto table = PerModuleInterface::GetInstance()->GetSystemTable();
    if (table == nullptr)
    {
        return {};
    }
    return table->getDefaultLogger();
}
