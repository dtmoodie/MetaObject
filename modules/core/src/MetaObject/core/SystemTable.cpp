#include "SystemTable.hpp"
#include "MetaObject/logging/logging.hpp"
#include "singletons.hpp"

std::weak_ptr<SystemTable> SystemTable::inst;

std::shared_ptr<SystemTable> SystemTable::instance()
{
    std::shared_ptr<SystemTable> output = SystemTable::inst.lock();
    if (!output)
    {
        output = std::make_shared<SystemTable>();
        SystemTable::inst = output;
    }
    return output;
}

bool SystemTable::checkInstance()
{
    auto inst = SystemTable::inst.lock();
    return static_cast<bool>(inst);
}

SystemTable::SystemTable()
{
    if (auto inst = SystemTable::inst.lock())
    {
        THROW(warning) << "Can only create one system table per process";
    }
}

SystemTable::~SystemTable()
{
}

void SystemTable::deleteSingleton(mo::TypeInfo type)
{
    g_singletons.erase(type);
}
