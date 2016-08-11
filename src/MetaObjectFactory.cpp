#include "MetaObject/MetaObjectFactory.hpp"
#include "RuntimeObjectSystem.h"
#include "IObjectFactorySystem.h"
#include "MetaObject/Logging/CompileLogger.hpp"
#include "MetaObject/IMetaObject.hpp"
using namespace mo;

struct MetaObjectFactory::impl
{
    impl(SystemTable* table)
    {
        obj_system.Initialise(&logger, table);
    }
    RuntimeObjectSystem obj_system;
    CompileLogger logger;
};

MetaObjectFactory::MetaObjectFactory(SystemTable* table)
{
    _pimpl = new impl(table);
}
MetaObjectFactory::~MetaObjectFactory()
{
    delete _pimpl;
}
IRuntimeObjectSystem* MetaObjectFactory::GetObjectSystem()
{
    return &_pimpl->obj_system;
}
MetaObjectFactory* MetaObjectFactory::Instance(SystemTable* table)
{
    static MetaObjectFactory* g_inst = nullptr;
    if(g_inst == nullptr)
    {
        g_inst = new MetaObjectFactory(table);
    }
    return g_inst;
}
IMetaObject* MetaObjectFactory::Create(const char* type_name)
{
    auto constructor = _pimpl->obj_system.GetObjectFactorySystem()->GetConstructor(type_name);
    if(constructor)
    {
        IObject* obj = constructor->Construct();
        if(IMetaObject* mobj = dynamic_cast<IMetaObject*>(obj))
        {
            mobj->Init(true);
            return mobj;
        }else
        {
            delete obj;
        }
    }
    return nullptr;
}