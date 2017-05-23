#pragma once
#include "IObjectInfo.h"
#include "MetaObject/detail/TypeInfo.hpp"
#include "MetaObject/detail/Export.hpp"

#include <vector>
namespace mo
{
    struct ParamInfo;
    struct SignalInfo;
    struct SlotInfo;
    class MO_EXPORTS IMetaObjectInfo: public IObjectInfo
    {
    public:
        virtual std::vector<ParamInfo*>     getParamInfo() const = 0;
        virtual std::vector<SignalInfo*>    getSignalInfo() const = 0;
        virtual std::vector<SlotInfo*>      getSlotInfo() const = 0;
        virtual TypeInfo                    getTypeInfo() const = 0;
        virtual std::string                 print() const;
        virtual std::string                 getDisplayName() const
        {
            return getObjectName();
        }
    };
}
