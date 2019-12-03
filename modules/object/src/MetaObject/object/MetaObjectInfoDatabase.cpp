#include "MetaObject/object/MetaObjectInfoDatabase.hpp"
#include "MetaObject/object/MetaObjectInfo.hpp"
using namespace mo;

MetaObjectInfoDatabase::MetaObjectInfoDatabase()
{
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

void MetaObjectInfoDatabase::registerInfo(IMetaObjectInfo* info_)
{
    info[info_->GetObjectName()] = info_;
}

std::vector<IMetaObjectInfo*> MetaObjectInfoDatabase::getMetaObjectInfo()
{
    std::vector<IMetaObjectInfo*> output;
    for (auto& itr : info)
    {
        output.push_back(itr.second);
    }
    return output;
}

IMetaObjectInfo* MetaObjectInfoDatabase::getMetaObjectInfo(std::string name)
{
    auto itr = info.find(name);
    if (itr != info.end())
    {
        return itr->second;
    }
    return nullptr;
}
