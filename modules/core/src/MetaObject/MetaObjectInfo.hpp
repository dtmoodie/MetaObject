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
        static void getParamInfoStatic(std::vector<ParamInfo*>& info)
        {
            T::getParamInfoStatic(info);
        }
        static void getSignalInfoStatic(std::vector<SignalInfo*>& info)
        {
            T::getSignalInfoStatic(info);
        }
        static void getSlotInfoStatic(std::vector<SlotInfo*>& info)
        {
            T::getSlotInfoStatic(info);
        }
		static std::string                 TooltipStatic()
        {
            return _get_tooltip_helper<T>();
        }
        static std::string                 DescriptionStatic()
        {
            return _get_description_helper<T>();
        }
        static TypeInfo                    getTypeInfoStatic()
        {
            return TypeInfo(typeid(typename T::BASE_CLASS));
        }
        std::vector<ParamInfo*>        getParamInfo() const
        {
            std::vector<ParamInfo*> info;
            getParamInfoStatic(info);
            return info;
        }
		std::vector<SignalInfo*>           getSignalInfo() const
        {
            std::vector<SignalInfo*> info;
            getSignalInfoStatic(info);
            return info;
        }
		std::vector<SlotInfo*>             getSlotInfo() const
        {
            std::vector<SlotInfo*> info;
            getSlotInfoStatic(info);
            return info;
        }
		std::string                        GetObjectTooltip() const
        {
            return TooltipStatic();
        }
		std::string                        GetObjectHelp() const
        {
            return DescriptionStatic();
        }
        TypeInfo                           getTypeInfo() const
        {
            return getTypeInfoStatic();
        }
        std::string                        GetObjectName() const
        {
            return T::GetTypeNameStatic();
        }
        unsigned int                       GetInterfaceId() const
        {
            return T::s_interfaceID;
        }
        virtual std::string GetInterfaceName() const
        {
            return T::GetInterfaceName();
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

