#pragma once
#include "IMetaObjectInfo.hpp"
#include "MetaObject/core/detail/Counter.hpp"
#include "MetaObject/core/detail/HelperMacros.hpp"
#include "MetaObject/core/detail/Placeholders.hpp"

#include "MetaObject/object/MetaObjectInfoDatabase.hpp"
#include "MetaObject/object/detail/MetaObjectMacros.hpp"

#include "MetaObject/params/TParamPtr.hpp"
#include "MetaObject/params/TInputParam.hpp"

#include <type_traits>

struct ISimpleSerializer;

namespace mo
{
    template<class T>
    struct TMetaObjectInterfaceHelper: public T
    {
        template<class DType, class ParamType>
        inline void operator()(const mo::Data<DType>& data, const mo::Name& name, const mo::Param<ParamType>& param, int32_t N, bool first_init)
        {
            param.get()->setName(name.get());
            T::addParam(param.get());
        }

        template<class DType>
        inline void operator()(const mo::Data<DType>& data, const mo::Name& name, const mo::Param<mo::TParamPtr<DType>>& param, int32_t N, bool first_init)
        {
            param.get()->setName(name.get());
            param.get()->updatePtr(data.get());
            T::addParam(param.get());
        }

        template<class DType>
        inline void operator()(const mo::Data<DType>& data, const mo::Name& name, const mo::Param<mo::TParamOutput<DType>>& param, int32_t N, bool first_init)
        {
            param.get()->setName(name.get());
            param.get()->updatePtr(data.get());
            T::addParam(param.get());
        }

        template<class DType>
        inline void operator()(const mo::Data<DType const *>& data, const mo::Name& name, const mo::Param<mo::TInputParamPtr<DType>>& param, int32_t N, bool first_init)
        {
            param.get()->setName(name.get());
            param.get()->setUserDataPtr(data.get());
            T::addParam(param.get());
        }

        void initParams(bool first_init) override
        {
            T::reflect(*this, mo::VisitationFilter<mo::INIT>(), mo::MemberFilter<mo::CONTROL>(), first_init);
            T::reflect(*this, mo::VisitationFilter<mo::INIT>(), mo::MemberFilter<mo::INPUT>(), first_init);
            T::reflect(*this, mo::VisitationFilter<mo::INIT>(), mo::MemberFilter<mo::OUTPUT>(), first_init);
            T::reflect(*this, mo::VisitationFilter<mo::INIT>(), mo::MemberFilter<mo::STATUS>(), first_init);
            T::reflect(*this, mo::VisitationFilter<mo::INIT>(), mo::MemberFilter<mo::STATE>(), first_init);
        }

        template<class AR>
        struct SerializeVisitor
        {
            AR& ar;
            template< class DType>
            inline void operator()(const mo::Data<DType>& data, const mo::Name& name)
            {
                ar(name.get(), data.get());
            }
        };

        template<class AR>
        void loadParams(AR& ar)
        {
            SerializeVisitor<AR> visitor{ar};
            T::reflect(visitor, mo::VisitationFilter<mo::SERIALIZE>(), mo::MemberFilter<mo::CONTROL>());
            T::reflect(visitor, mo::VisitationFilter<mo::SERIALIZE>(), mo::MemberFilter<mo::STATE>());
        }

        template<class AR>
        void saveParams(AR& ar)
        {
            SerializeVisitor<AR> visitor{ar};
            T::reflect(visitor, mo::VisitationFilter<mo::SERIALIZE>(), mo::MemberFilter<mo::CONTROL>());
            T::reflect(visitor, mo::VisitationFilter<mo::SERIALIZE>(), mo::MemberFilter<mo::STATE>());
        }

        void serializeParams(ISimpleSerializer* serializer) override
        {

        }

        template<class AR>
        void save( AR& ar ) const
        {

        }

        template<class AR>
        void load( AR& ar )
        {

        }


        template<class Sig, class R, class T1, class... Args>
        inline void operator()(const mo::Slot<Sig>& data, const mo::NamedType<R(T1::*)(Args...), Function>& fptr, const mo::Name& name, int N, bool first_init)
        {
            // TODO finish slot initialization
            (*data.get()) = variadicBind(fptr.get(), static_cast<T*>(this), make_int_sequence<sizeof...(Args)>{});
            this->addSlot(data.get(), name.get());
        }

        void bindSlots(bool first_init) override
        {
            T::reflect(*this, mo::VisitationFilter<mo::INIT>(), mo::MemberFilter<mo::SLOTS>(), first_init);
        }

        template<class Sig>
        inline void operator()(const mo::Signal<Sig>& data, const mo::Name& name, int N, bool first_init, int32_t* counter)
        {
            // TODO finish signal initialization
        }

        int initSignals(bool first_init) override
        {
            int32_t count = 0;
            T::reflect(*this, mo::VisitationFilter<mo::INIT>(), mo::MemberFilter<mo::SIGNALS>(), first_init,  &count);
            return count;
        }

        struct ParamInfoVisitor
        {
            template<class DType, class ParamType, int32_t N>
            inline void operator()(const mo::Data<DType>& data, const mo::Name& name, const mo::Param<ParamType>& param, mo::_counter_<N>)
            {

            }
            std::vector<mo::ParamInfo*>& vec;
        };


        static void getParamInfoStatic(std::vector<mo::ParamInfo*>& vec)
        {
            ParamInfoVisitor visitor{vec};
            static_cast<T*>(nullptr)->reflect(visitor,
                       mo::VisitationFilter<mo::LIST>(), mo::MemberFilter<mo::CONTROL>());
        }

        static std::vector<mo::ParamInfo*> getParamInfoStatic()
        {
            std::vector<mo::ParamInfo*> vec;
            getParamInfoStatic(vec);
            return vec;
        }

        static void getSignalInfoStatic(std::vector<mo::SignalInfo*>& vec)
        {

        }

        static std::vector<mo::SignalInfo*> getSignalInfoStatic()
        {
            std::vector<mo::SignalInfo*> vec;
            getSignalInfoStatic(vec);
            return vec;
        }

        static void getSlotInfoStatic(std::vector<mo::SlotInfo*>& vec)
        {

        }

        static std::vector<mo::SlotInfo*> getSlotInfoStatic()
        {
            std::vector<mo::SlotInfo*> vec;
            getSlotInfoStatic(vec);
            return vec;
        }

        void getParamInfo(std::vector<mo::ParamInfo*>& vec) const override
        {
            getParamInfoStatic(vec);
        }

        void getSignalInfo(std::vector<mo::SignalInfo*>& vec) const override
        {
            getSignalInfoStatic(vec);
        }

        void getSlotInfo(std::vector<mo::SlotInfo*>& vec) const override
        {
            getSlotInfoStatic(vec);
        }



    };

    // Static object information available for each meta object
    // Used for static introspection
    // Specialize this for each class which requires additional fields
    template <class Type, class InterfaceInfo>
    struct MetaObjectInfoImpl : public InterfaceInfo
    {
    };

    template <class T>
    struct MetaObjectInfo : public MetaObjectInfoImpl<T, typename T::InterfaceInfo>
    {
        MetaObjectInfo() { MetaObjectInfoDatabase::instance()->registerInfo(this); }

        static void getParamInfoStatic(std::vector<ParamInfo*>& info) { T::getParamInfoStatic(info); }

        static void getSignalInfoStatic(std::vector<SignalInfo*>& info) { T::getSignalInfoStatic(info); }

        static void getSlotInfoStatic(std::vector<SlotInfo*>& info) { T::getSlotInfoStatic(info); }

        static std::string tooltipStatic() { return _get_tooltip_helper<T>(); }

        static std::string descriptionStatic() { return _get_description_helper<T>(); }

        static TypeInfo getTypeInfoStatic() { return TypeInfo(typeid(typename T::BASE_CLASS)); }

        std::vector<ParamInfo*> getParamInfo() const
        {
            std::vector<ParamInfo*> info;
            getParamInfoStatic(info);
            return info;
        }

        std::vector<SignalInfo*> getSignalInfo() const
        {
            std::vector<SignalInfo*> info;
            getSignalInfoStatic(info);
            return info;
        }

        std::vector<SlotInfo*> getSlotInfo() const
        {
            std::vector<SlotInfo*> info;
            getSlotInfoStatic(info);
            return info;
        }

        std::string getObjectTooltip() const { return tooltipStatic(); }

        std::string getObjectHelp() const { return descriptionStatic(); }

        TypeInfo getTypeInfo() const { return getTypeInfoStatic(); }

        std::string GetObjectName() const { return T::GetTypeNameStatic(); }

        unsigned int GetInterfaceId() const { return T::getHash(); }

        virtual std::string GetInterfaceName() const { return T::GetInterfaceName(); }

        virtual IObjectConstructor* GetConstructor() const { return T::GetConstructorStatic(); }

        virtual bool InheritsFrom(InterfaceID iid) const override { return T::InheritsFrom(iid); }
      private:
        DEFINE_HAS_STATIC_FUNCTION(HasTooltip, getTooltipStatic, std::string (*)(void));
        DEFINE_HAS_STATIC_FUNCTION(HasDescription, getDescriptionStatic, std::string (*)(void));
        template <class U>
        static std::string _get_tooltip_helper(typename std::enable_if<HasTooltip<U>::value, void>::type* = 0)
        {
            return U::getTooltipStatic();
        }
        template <class U>
        static std::string _get_tooltip_helper(typename std::enable_if<!HasTooltip<U>::value, void>::type* = 0)
        {
            return "";
        }
        template <class U>
        static std::string _get_description_helper(typename std::enable_if<HasDescription<U>::value, void>::type* = 0)
        {
            return U::getDescriptionStatic();
        }
        template <class U>
        static std::string _get_description_helper(typename std::enable_if<!HasDescription<U>::value, void>::type* = 0)
        {
            return "";
        }
    };
}
