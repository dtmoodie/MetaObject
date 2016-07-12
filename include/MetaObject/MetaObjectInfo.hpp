#pragma once
#include <vector>
namespace mo
{
	// Static object information available for each meta object
	// Used for static introspection
	struct ParameterInfo;
	struct SignalInfo;
	struct SlotInfo;
	struct IMetaObjectInfo
	{
		virtual std::vector<ParameterInfo*> ListParameters();
		virtual std::vector<SignalInfo*> ListSignalInfo();
		virtual std::vector<SlotInfo*> ListSlotInfo();
		virtual std::string Tooltip();
		virtual std::string Description();
	};
}

