#pragma once
#include "MetaObject/Detail/Export.hpp"

namespace mo
{
    class IMetaObjectInfo;
    class MO_EXPORTS MetaObjectInfoDatabase
    {
    public:
        static MetaObjectInfoDatabase* Instance();
        
        void RegisterInfo(IMetaObjectInfo* info);
        
        std::vector<IMetaObjectInfo*> GetMetaObjectInfo();
        IMetaObjectInfo* GetMetaObjectInfo(std::string name);
    };
}