#pragma once
#include "IMetaObjectInfo.hpp"
#include "MetaObject/Detail/Counter.hpp"
#include "MetaObject/MetaObjectInfoDatabase.hpp"



namespace mo
{
	// Static object information available for each meta object
	// Used for static introspection
    template<class T, int N, typename Enable = void> 
    struct MetaObjectInfo: virtual public IMetaObjectInfo
    {
        MetaObjectInfo()
        {
            MetaObjectInfoDatabase::Instance()->RegisterInfo(this);   
        }
        static std::vector<ParameterInfo*> ListParametersStatic()
        {
            std::vector<ParameterInfo*> info;
            T::list_parameter_info_(info, mo::_counter_<N>());
            return info;
        }
        static std::vector<SignalInfo*>    ListSignalInfoStatic()
        {
            std::vector<SignalInfo*> info; 
            T::list_signal_info_(info, mo::_counter_<N>());
            return info;
        }
        static std::vector<SlotInfo*>      ListSlotInfoStatic()
        {
            std::vector<SlotInfo*> info;
            T::list_slots_(info, mo::_counter_<N>());
            return info;
        }
		static std::string                 TooltipStatic()
        {
            return "";
        }
        static std::string                 DescriptionStatic()
        {
            return "";
        }
        static TypeInfo                    GetTypeInfoStatic()
        {
            return TypeInfo(typeid(typename T::BASE_CLASS));
        }
        std::vector<ParameterInfo*>        ListParameters() const
        {
            return ListParametersStatic();
        }
		std::vector<SignalInfo*>           ListSignalInfo() const
        {
            return ListSignalInfoStatic();
        }
		std::vector<SlotInfo*>             ListSlotInfo() const
        {
            return ListSlotInfoStatic();
        }
		std::string                        GetObjectTooltip() const
        {
            return TooltipStatic();
        }
		std::string                        GetObjectHelp() const
        {
            return DescriptionStatic();
        }
        TypeInfo                           GetTypeInfo() const
        {
            return GetTypeInfoStatic();
        }
        std::string                        GetObjectName() const
        {
            return T::GetTypeNameStatic();
        }
        int                                GetInterfaceId() const
        {
            return T::s_interfaceID;
        }
    };
}

