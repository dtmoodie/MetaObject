#pragma once
#include "MetaObject/Detail/TypeInfo.h"
#include "IObjectInfo.h"
#include <vector>

namespace mo
{
	// Static object information available for each meta object
	// Used for static introspection
	struct ParameterInfo;
	struct SignalInfo;
	struct SlotInfo;
	struct IMetaObjectInfo: IObjectInfo
	{
		virtual std::vector<ParameterInfo*> ListParameters() = 0;
		virtual std::vector<SignalInfo*>    ListSignalInfo() = 0;
		virtual std::vector<SlotInfo*>      ListSlotInfo() = 0;
		virtual std::string                 Tooltip() = 0;
		virtual std::string                 Description() = 0;
        virtual TypeInfo                    GetTypeInfo() = 0;
	};
}

