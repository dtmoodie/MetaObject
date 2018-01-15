#include "SystemTable.hpp"
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

void SystemTable::set(std::weak_ptr<SystemTable> table)
{
    SystemTable::inst = table;
}

SystemTable::SystemTable()
{
    
}

SystemTable::~SystemTable()
{

}

void SystemTable::deleteSingleton(mo::TypeInfo type)
{
    g_singletons.erase(type);
}

