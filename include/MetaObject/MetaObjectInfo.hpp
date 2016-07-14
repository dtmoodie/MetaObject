#pragma once
#include "MetaObject/Detail/TypeInfo.h"
#include "MetaObject/Detail/Counter.hpp"
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/MetaObjectInfoDatabase.hpp"
#include "IObjectInfo.h"
#include <vector>

namespace mo
{
	// Static object information available for each meta object
	// Used for static introspection
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
		virtual std::string                 Tooltip() = 0;
		virtual std::string                 Description() = 0;
        virtual TypeInfo                    GetTypeInfo() = 0;
        virtual std::string                 GetTypeName() = 0;
        virtual std::string                 GetDisplayName() = 0;
	};

    template<class T, int N> struct MetaObjectInfo: public IObjectInfo
    {
        MetaObjectInfo()
        {
            MetaObjectInfoDatabase::Instance()->RegisterInfo(this);   
        }
        static std::vector<ParameterInfo*> ListParametersStatic()
        {
            return T::list_parameters_(mo::_counter_<N>());
        }

		static std::vector<SignalInfo*>    ListSignalInfoStatic()
        {
            return T::list_signals_(mo::_counter_<N>());
        }

		static std::vector<SlotInfo*>      ListSlotInfoStatic()
        {
            return T::list_slots_(mo::_counter_<N>());
        }

        static std::vector<CallbackInfo*> ListCallbackInfoStatic()
        {
            return T::list_callbacks_(mo::_counter_<N>());
        }

		static std::string                 TooltipStatic()
        {
            return T::Tooltip();
        }

		static std::string                 DescriptionStatic()
        {
            return T::Description();
        }

        static TypeInfo                    GetTypeInfoStatic()
        {
            return TypeInfo(typeid(typename T::BASE_CLASS));
        }

        std::vector<ParameterInfo*> ListParameters()
        {
            return ListParametersStatic();
        }
		std::vector<SignalInfo*>    ListSignalInfo()
        {
            return ListSignalInfoStatic();
        }
		std::vector<SlotInfo*>      ListSlotInfo()
        {
            return ListSlotInfoStatic();
        }
        std::vector<CallbackInfo*>  ListCallbackInfo()
        {
            return ListCallbackInfoStatic();
        }
		std::string                 Tooltip()
        {
            return TooltipStatic();
        }
		std::string                 Description()
        {
            return DescriptionStatic();
        }
        TypeInfo                    GetTypeInfo()
        {
            return GetTypeInfoStatic();
        }
        std::string                 GetTypeName()
        {
            return T::GetTypeNameStatic();
        }
    };
}

