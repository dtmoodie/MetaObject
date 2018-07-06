#pragma once
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
#include "RuntimeObjectSystem/IObjectInfo.h"

#include <vector>
namespace mo
{
    struct ParamInfo;
    struct SignalInfo;
    struct SlotInfo;
    struct ISlot;
    class MO_EXPORTS IMetaObjectInfo : public IObjectInfo
    {
      public:
        virtual std::vector<ParamInfo*> getParamInfo() const = 0;
        virtual std::vector<SignalInfo*> getSignalInfo() const = 0;
        virtual std::vector<SlotInfo*> getSlotInfo() const = 0;
        virtual TypeInfo getTypeInfo() const = 0;
        virtual std::string Print(IObjectInfo::Verbosity verbosity = IObjectInfo::INFO) const; // Pascal case in RCC
        virtual std::string getObjectTooltip() const;
        virtual std::string getObjectHelp() const;
        virtual std::string getDisplayName() const { return GetObjectName(); }
        virtual std::vector<std::pair<ISlot*, std::string>> getStaticSlots() const { return {}; }
    };
}
