#pragma once
#include "IMetaObjectInfo.hpp"
#include "MetaObject/core/detail/HelperMacros.hpp"
#include "MetaObject/core/detail/Counter.hpp"
#include "MetaObject/object/MetaObjectInfoDatabase.hpp"
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
            MetaObjectInfoDatabase::instance()->registerInfo(this);
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

        static std::string                 tooltipStatic()
        {
            return _get_tooltip_helper<T>();
        }

        static std::string                 descriptionStatic()
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

        std::string                        getObjectTooltip() const
        {
            return tooltipStatic();
        }

        std::string                        getObjectHelp() const
        {
            return descriptionStatic();
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

        virtual IObjectConstructor* GetConstructor() const
        {
           return T::GetConstructorStatic();
        }

        virtual bool InheritsFrom(InterfaceID iid) const override
        {
            return T::InheritsFrom(iid);
        }
    private:
        DEFINE_HAS_STATIC_FUNCTION(HasTooltip, getTooltipStatic, std::string(*)(void));
        DEFINE_HAS_STATIC_FUNCTION(HasDescription, getDescriptionStatic, std::string(*)(void));
        template<class U> static std::string _get_tooltip_helper(typename std::enable_if<HasTooltip<U>::value, void>::type* = 0)
        {
            return U::getTooltipStatic();
        }
        template<class U> static std::string _get_tooltip_helper(typename std::enable_if<!HasTooltip<U>::value, void>::type* = 0)
        {
            return "";
        }
        template<class U> static std::string _get_description_helper(typename std::enable_if<HasDescription<U>::value, void>::type* = 0)
        {
            return U::getDescriptionStatic();
        }
        template<class U> static std::string _get_description_helper(typename std::enable_if<!HasDescription<U>::value, void>::type* = 0)
        {
            return "";
        }
    };
}

