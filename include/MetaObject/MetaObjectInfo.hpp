#pragma once
#include "IMetaObjectInfo.hpp"
#include "MetaObject/Detail/HelperMacros.hpp"
#include "MetaObject/Detail/Counter.hpp"
#include "MetaObject/MetaObjectInfoDatabase.hpp"
#include <type_traits>



namespace mo
{
	// Static object information available for each meta object
	// Used for static introspection

    // Specialize this for each class which requires additional fields
    template<class Type, class InterfaceInfo>
    struct MetaObjectInfoImpl: public InterfaceInfo
    {
    };

    template<class T> 
    struct MetaObjectInfo: public MetaObjectInfoImpl<T, typename T::InterfaceInfo>
    {
        MetaObjectInfo()
        {
            MetaObjectInfoDatabase::Instance()->RegisterInfo(this);   
        }
        static std::vector<ParameterInfo*> ListParametersStatic()
        {
            return T::GetParameterInfoStatic();
        }
        static std::vector<SignalInfo*>    ListSignalInfoStatic()
        {
            return T::GetSignalInfoStatic();
        }
        static std::vector<SlotInfo*>      ListSlotInfoStatic()
        {
            return T::GetSlotInfoStatic();
        }
		static std::string                 TooltipStatic()
        {
            return _get_tooltip_helper<T>();
        }
        static std::string                 DescriptionStatic()
        {
            return _get_description_helper<T>();
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
    private:
        DEFINE_HAS_STATIC_FUNCTION(HasTooltip, V::GetTooltipStatic, std::string(*)(void));
        DEFINE_HAS_STATIC_FUNCTION(HasDescription, V::GetDescriptionStatic, std::string(*)(void));
        template<class U> static std::string _get_tooltip_helper(typename std::enable_if<HasTooltip<U>::value, void>::type* = 0)
        {
            return U::GetTooltipStatic();
        }
        template<class U> static std::string _get_tooltip_helper(typename std::enable_if<!HasTooltip<U>::value, void>::type* = 0)
        {
            return "";
        }
        template<class U> static std::string _get_description_helper(typename std::enable_if<HasDescription<U>::value, void>::type* = 0)
        {
            return U::GetDescriptionStatic();
        }
        template<class U> static std::string _get_description_helper(typename std::enable_if<!HasDescription<U>::value, void>::type* = 0)
        {
            return "";
        }
    };
}

