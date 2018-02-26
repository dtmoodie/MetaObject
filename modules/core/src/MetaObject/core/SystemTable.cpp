#include "SystemTable.hpp"
#include "MetaObject/logging/logging.hpp"
#include "singletons.hpp"

std::weak_ptr<SystemTable> inst;

std::shared_ptr<SystemTable> SystemTable::instance()
{
    std::shared_ptr<SystemTable> output = inst.lock();
    if (!output)
    {
        output = std::make_shared<SystemTable>();
        inst = output;
    }
    return output;
}

void SystemTable::setInstance(const std::shared_ptr<SystemTable>& table)
{
    auto current = inst.lock();
    MO_ASSERT(!current);
    inst = table;
}

bool SystemTable::checkInstance()
{
    auto inst_ = inst.lock();
    return static_cast<bool>(inst_);
}

SystemTable::SystemTable()
{
    if (auto inst_ = inst.lock())
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
