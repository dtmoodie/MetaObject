#include "MetaObject/object/MetaObjectInfoDatabase.hpp"
#include "MetaObject/object/MetaObjectInfo.hpp"
#include <map>
using namespace mo;

struct MetaObjectInfoDatabase::impl
{
    std::map<std::string, IMetaObjectInfo*> info;
};

MetaObjectInfoDatabase::MetaObjectInfoDatabase()
{
    _pimpl = new impl();
}
MetaObjectInfoDatabase::~MetaObjectInfoDatabase()
{
    delete _pimpl;
}

MetaObjectInfoDatabase* MetaObjectInfoDatabase::instance()
{
    static MetaObjectInfoDatabase* g_inst = nullptr;
    if (g_inst == nullptr)
    {
        g_inst = new MetaObjectInfoDatabase();
    }
    return g_inst;
}

void MetaObjectInfoDatabase::registerInfo(IMetaObjectInfo* info)
{
    _pimpl->info[info->GetObjectName()] = info;
}

std::vector<IMetaObjectInfo*> MetaObjectInfoDatabase::getMetaObjectInfo()
{
    std::vector<IMetaObjectInfo*> output;
    for (auto& itr : _pimpl->info) {
        output.push_back(itr.second);
    }
    return output;
}

IMetaObjectInfo* MetaObjectInfoDatabase::getMetaObjectInfo(std::string name)
{
    auto itr = _pimpl->info.find(name);
    if (itr != _pimpl->info.end())
    {
        return itr->second;
    }
    return nullptr;
}
