#pragma once
#include "IObjectInfo.h"
#include "MetaObject/Detail/TypeInfo.h"
#include "MetaObject/Detail/Export.hpp"

#include <vector>
namespace mo
{
    struct ParameterInfo;
	struct SignalInfo;
	struct SlotInfo;
	struct MO_EXPORTS IMetaObjectInfo: IObjectInfo
	{
		virtual std::vector<ParameterInfo*> ListParameters() const = 0;
		virtual std::vector<SignalInfo*>    ListSignalInfo() const = 0;
		virtual std::vector<SlotInfo*>      ListSlotInfo() const = 0;
        virtual TypeInfo                    GetTypeInfo() const = 0;
        virtual std::string                 Print() const;
        virtual std::string                 GetDisplayName() const
        {
            return GetObjectName();
        }
	};
}