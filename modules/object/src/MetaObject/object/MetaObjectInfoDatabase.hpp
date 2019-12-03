#pragma once
#include "MetaObject/detail/Export.hpp"
#include <memory>
#include <string>
#include <vector>
#include <map>
namespace mo
{
    class IMetaObjectInfo;
    class MO_EXPORTS MetaObjectInfoDatabase
    {
      public:
        static MetaObjectInfoDatabase* instance();

        void registerInfo(IMetaObjectInfo* info);

        std::vector<IMetaObjectInfo*> getMetaObjectInfo();
        IMetaObjectInfo* getMetaObjectInfo(std::string name);

      private:
        MetaObjectInfoDatabase();
        std::map<std::string, IMetaObjectInfo*> info;
    };
}
