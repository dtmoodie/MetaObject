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
} // namespace mo

spdlog::details::registry& mo::getLoggerRegistry()
{
    auto table = PerModuleInterface::GetInstance()->GetSystemTable();
    if (table == nullptr)
    {
        throw std::runtime_error("SystemTable == nullptr");
    }

    return getLoggerRegistry(*table);
}

spdlog::details::registry& mo::getLoggerRegistry(SystemTable& table)
{
    auto registry = table.getSingleton<spdlog::details::registry>();
    return *registry;
}

spdlog::logger& mo::getDefaultLogger()
{
    PerModuleInterface* instance = PerModuleInterface::GetInstance();
    auto table = instance->GetSystemTable();
    if (table == nullptr)
    {
        throw std::runtime_error("SystemTable == nullptr");
    }
    auto logger = table->getLogger();
    if (logger == nullptr)
    {
        throw std::runtime_error("table->getDefaultLogger() == nullptr");
    }
    return *logger;
}

std::shared_ptr<spdlog::logger> mo::getLogger(const std::string& name)
{
    auto table = PerModuleInterface::GetInstance()->GetSystemTable();
    if (table == nullptr)
    {
        return {};
    }
    return table->getLogger(name);
}
