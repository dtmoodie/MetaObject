#pragma once
#include "IMetaObjectInfo.hpp"
#include "MetaObject/Detail/Counter.hpp"
#include "MetaObject/MetaObjectInfoDatabase.hpp"



namespace mo
{
	// Static object information available for each meta object
	// Used for static introspection
    template<class T, int N> struct MetaObjectInfo: public IMetaObjectInfo
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
        static std::vector<CallbackInfo*>  ListCallbackInfoStatic()
        {
            std::vector<CallbackInfo*> info;
            T::list_callbacks_(info, mo::_counter_<N>());
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
        std::vector<ParameterInfo*>        ListParameters()
        {
            return ListParametersStatic();
        }
		std::vector<SignalInfo*>           ListSignalInfo()
        {
            return ListSignalInfoStatic();
        }
		std::vector<SlotInfo*>             ListSlotInfo()
        {
            return ListSlotInfoStatic();
        }
        std::vector<CallbackInfo*>         ListCallbackInfo()
        {
            return ListCallbackInfoStatic();
        }
		std::string                        GetObjectTooltip()
        {
            return TooltipStatic();
        }
		std::string                        GetObjectHelp()
        {
            return DescriptionStatic();
        }
        TypeInfo                           GetTypeInfo()
        {
            return GetTypeInfoStatic();
        }
        std::string                        GetObjectName()
        {
            return T::GetTypeNameStatic();
        }
        int                                GetInterfaceId()
        {
            return T::s_interfaceID;
        }
    };
}

