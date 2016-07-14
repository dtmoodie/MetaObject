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
    struct CallbackInfo;
	struct MO_EXPORTS IMetaObjectInfo: IObjectInfo
	{
		virtual std::vector<ParameterInfo*> ListParameters() = 0;
		virtual std::vector<SignalInfo*>    ListSignalInfo() = 0;
		virtual std::vector<SlotInfo*>      ListSlotInfo() = 0;
        virtual std::vector<CallbackInfo*>  ListCallbackInfo() = 0;
        virtual TypeInfo                    GetTypeInfo() = 0;
        virtual std::string                 Print();
        virtual std::string                 GetDisplayName()
        {
            return GetObjectName();
        }
	};
}