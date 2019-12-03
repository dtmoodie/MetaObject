#ifndef MO_OBJECT_META_OBJECT_INFO_HPP
#define MO_OBJECT_META_OBJECT_INFO_HPP

#include "IMetaObjectInfo.hpp"

#include "MetaObject/core/detail/HelperMacros.hpp"

#include "MetaObject/object/MetaObjectInfoDatabase.hpp"
#include "MetaObject/object/detail/MetaObjectMacros.hpp"

#include "MetaObject/params/ParamInfo.hpp"
#include "MetaObject/params/TInputParam.hpp"
#include "MetaObject/params/TParamPtr.hpp"
#include <MetaObject/params/TParamOutput.hpp>

#include "MetaObject/signals/SignalInfo.hpp"
#include "MetaObject/signals/SlotInfo.hpp"

#include <RuntimeObjectSystem/ISimpleSerializer.h>

#include <ct/Indexer.hpp>
#include <ct/VariadicTypedef.hpp>
#include <ct/reflect/metadata.hpp>
#include <ct/reflect/visitor.hpp>

#include <type_traits>

namespace mo
{
    template <class... T>
    class TMultiInput;
    template <class... T>
    class TMultiOutput;

    template <class T>
    struct LoadSerializer
    {
        template <class DTYPE, class CTYPE, ct::Flag_t FLAGS, class METADATA>
        auto serialize(const ct::MemberObjectPointer<DTYPE CTYPE::*, FLAGS, METADATA> ptr)
            -> ct::EnableIf<!std::is_const<typename std::remove_pointer<DTYPE>::type>::value>
        {
            m_visitor(&ptr.set(*m_this), ptr.m_name);
        }

        template <class DTYPE, class CTYPE, ct::Flag_t FLAGS, class METADATA>
        auto serialize(const ct::MemberObjectPointer<DTYPE CTYPE::*, FLAGS, METADATA>)
            -> ct::EnableIf<std::is_const<typename std::remove_pointer<DTYPE>::type>::value>
        {
        }
        ILoadVisitor& m_visitor;
        T* m_this;
    };

    template <class T>
    struct SaveSerializer
    {
        template <class DTYPE, class CTYPE, ct::Flag_t FLAGS, class METADATA>
        auto serialize(const ct::MemberObjectPointer<DTYPE CTYPE::*, FLAGS, METADATA> ptr)
            -> ct::EnableIf<!std::is_const<typename std::remove_pointer<DTYPE>::type>::value>
        {
            m_visitor(&ptr.get(*m_this), ptr.m_name);
        }

        template <class DTYPE, class CTYPE, ct::Flag_t FLAGS, class METADATA>
        auto serialize(const ct::MemberObjectPointer<DTYPE CTYPE::*, FLAGS, METADATA>)
            -> ct::EnableIf<std::is_const<typename std::remove_pointer<DTYPE>::type>::value>
        {
        }
        ISaveVisitor& m_visitor;
        const T* m_this;
    };

    template <class T>
    struct SerializerWrapper
    {
        template <class PTR>
        void serialize(const PTR ptr)
        {
            m_serializer->SerializeProperty(ptr.getName().cStr(), ptr.set(*m_this));
        }

        ISimpleSerializer* m_serializer;
        T* m_this;
    };

    template <class T>
    struct TMetaObjectInterfaceHelper : public T
    {
        template <class U1, class U2, ct::Flag_t F, class M>
        void initParam(ct::MemberPropertyPointer<U1, U2, F, M>, ct::index_t)
        {
        }

        template <class U, ct::Flag_t FLAGS, class METADATA, class... PTRS, ct::index_t I>
        void initParam(ct::StaticFunctions<U, FLAGS, METADATA, PTRS...>, ct::Indexer<I>)
        {
        }

        template <class U, ct::Flag_t FLAGS, class METADATA, class... PTRS, ct::index_t I>
        void initParam(ct::MemberFunctionPointers<U, FLAGS, METADATA, PTRS...>, ct::Indexer<I>)
        {
        }

        template <class DTYPE, class CTYPE, ct::Flag_t FLAGS, class METADATA, ct::index_t I>
        void initParam(ct::MemberObjectPointer<mo::TInputParamPtr<DTYPE> CTYPE::*, FLAGS, METADATA> ptr, ct::Indexer<I>)
        {
            constexpr const ct::index_t J = ct::indexOfField<T>(ct::getName<I, T>().slice(0, -6));
            ct::StaticInequality<ct::index_t, J, -1>{};
            auto wrapped_field_ptr = ct::Reflect<T>::getPtr(Indexer<J>{});
            auto param_ptr = &ptr.set(*this);
            param_ptr->setName(ct::getName<I, T>().slice(0, -6));
            param_ptr->setUserDataPtr(&wrapped_field_ptr.set(*this));
            T::addParam(param_ptr);
        }

        template <class DTYPE, class CTYPE, uint64_t PARAM_FLAGS, ct::Flag_t FLAGS, class METADATA, ct::index_t I>
        void initParam(ct::MemberObjectPointer<mo::TParamOutput<DTYPE, PARAM_FLAGS> CTYPE::*, FLAGS, METADATA> ptr,
                       ct::Indexer<I>)
        {
            auto param_ptr = &ptr.set(*this);
            param_ptr->setName(ct::getName<I, T>());
            auto initializer = ct::metadata<ct::Initializer<DTYPE>>(ptr);
            if (initializer)
            {
                param_ptr->updateData(initializer->getInitialValue());
            }
            T::addParam(param_ptr);
        }

        template <class DTYPE, class CTYPE, ct::Flag_t FLAGS, class METADATA, ct::index_t I>
        void initParam(ct::MemberObjectPointer<mo::TParamPtr<DTYPE> CTYPE::*, FLAGS, METADATA> ptr, ct::Indexer<I>)
        {
            constexpr const ct::index_t J = ct::indexOfField<T>(ct::getName<I, T>().slice(0, -6));
            ct::StaticInequality<ct::index_t, J, -1>{};
            auto wrapped_field_ptr = ct::Reflect<T>::getPtr(Indexer<J>{});
            auto param_ptr = &ptr.set(*this);
            param_ptr->setName(ct::getName<I, T>().slice(0, -6));
            param_ptr->updatePtr(&wrapped_field_ptr.set(*this), false);
            auto initializer = ct::metadata<ct::Initializer<DTYPE>>(wrapped_field_ptr);
            if (initializer)
            {
                wrapped_field_ptr.set(*this) = initializer->getInitialValue();
            }
            T::addParam(param_ptr);
        }

        template <class DTYPE, class CTYPE, ct::Flag_t FLAGS, class METADATA, ct::index_t I>
        void initParam(ct::MemberObjectPointer<DTYPE CTYPE::*, FLAGS, METADATA>, ct::Indexer<I>)
        {
            // ignore for now, added in the above initParam for TParamPtr
        }

        void initParamsRecurse(const bool, const ct::Indexer<0> idx)
        {
            const auto ptr = ct::Reflect<T>::getPtr(idx);
            initParam(ptr, idx);
        }

        template <ct::index_t I>
        void initParamsRecurse(const bool first_init, const ct::Indexer<I> idx)
        {
            const auto ptr = ct::Reflect<T>::getPtr(idx);
            initParam(ptr, idx);
            initParamsRecurse(first_init, --idx);
        }

        void initParams(bool first_init) override
        {
            initParamsRecurse(first_init, ct::Reflect<T>::end());
        }

        template <class SERIALIZER, class PTR, index_t I>
        void serializeParam(SERIALIZER&, PTR, ct::Indexer<I>) const
        {
        }

        template <class SERIALIZER, class DTYPE, class CTYPE, ct::Flag_t FLAGS, class METADATA, ct::index_t I>
        void serializeParam(SERIALIZER&,
                            ct::MemberObjectPointer<mo::TParamPtr<DTYPE> CTYPE::*, FLAGS, METADATA>,
                            ct::Indexer<I>) const
        {
        }

        template <class SERIALIZER, class DTYPE, class CTYPE, ct::Flag_t FLAGS, class METADATA, ct::index_t I>
        void serializeParam(SERIALIZER&,
                            ct::MemberObjectPointer<mo::TInputParamPtr<DTYPE> CTYPE::*, FLAGS, METADATA>,
                            ct::Indexer<I>) const
        {
        }

        template <class SERIALIZER,
                  class DTYPE,
                  class CTYPE,
                  mo::ParamFlags::EnumValueType PARAM_FLAGS,
                  ct::Flag_t FLAGS,
                  class METADATA,
                  ct::index_t I>
        void serializeParam(SERIALIZER&,
                            ct::MemberObjectPointer<mo::TParamOutput<DTYPE, PARAM_FLAGS> CTYPE::*, FLAGS, METADATA>,
                            ct::Indexer<I>) const
        {
        }

        template <class SERIALIZER, class DTYPE, class CTYPE, ct::Flag_t FLAGS, class METADATA, ct::index_t I>
        void serializeParam(SERIALIZER&,
                            ct::MemberObjectPointer<mo::TSignal<DTYPE> CTYPE::*, FLAGS, METADATA>,
                            ct::Indexer<I>) const
        {
        }

        template <class SERIALIZER, class DTYPE, class CTYPE, ct::Flag_t FLAGS, class METADATA, ct::index_t I>
        void serializeParam(SERIALIZER&,
                            ct::MemberObjectPointer<mo::TSlot<DTYPE> CTYPE::*, FLAGS, METADATA>,
                            ct::Indexer<I>) const
        {
        }

        template <class SERIALIZER, class DTYPE, class CTYPE, ct::Flag_t FLAGS, class METADATA, ct::index_t I>
        void serializeParam(SERIALIZER& serializer,
                            ct::MemberObjectPointer<DTYPE CTYPE::*, FLAGS, METADATA> ptr,
                            ct::Indexer<I>) const
        {
            serializer.serialize(ptr);
        }

        template <class SERIALIZER>
        void serializeParamsRecurse(SERIALIZER& serializer, ct::Indexer<0> idx) const
        {
            serializeParam(serializer, ct::Reflect<T>::getPtr(idx), idx);
        }

        template <class SERIALIZER, index_t I>
        void serializeParamsRecurse(SERIALIZER& serializer, ct::Indexer<I> idx) const
        {
            serializeParam(serializer, ct::Reflect<T>::getPtr(idx), idx);
            serializeParamsRecurse(serializer, --idx);
        }

        void load(ILoadVisitor& visitor) override
        {
            LoadSerializer<T> wrapper{visitor, this};
            serializeParamsRecurse(wrapper, ct::Reflect<T>::end());
        }

        void save(ISaveVisitor& visitor) const override
        {
            SaveSerializer<T> wrapper{visitor, this};
            serializeParamsRecurse(wrapper, ct::Reflect<T>::end());
        }

        void serializeParams(ISimpleSerializer* serializer) override
        {
            SerializerWrapper<T> wrapper{serializer, this};
            serializeParamsRecurse(wrapper, ct::Reflect<T>::end());
        }

        template <class PTR, ct::index_t I>
        static void paramInfo(std::vector<mo::ParamInfo*>&, PTR, ct::Indexer<I>)
        {
        }

        template <class DTYPE, class CTYPE, ct::Flag_t FLAGS, class METADATA, ct::index_t I>
        static void paramInfo(std::vector<mo::ParamInfo*>& vec,
                              ct::MemberObjectPointer<mo::TInputParamPtr<DTYPE> CTYPE::*, FLAGS, METADATA> ptr,
                              ct::Indexer<I>)
        {
            static mo::ParamInfo param_info(
                mo::TypeInfo(typeid(DTYPE)), ptr.m_name.slice(0, -6).toString(), "", "", mo::ParamFlags::kINPUT, "");
            vec.push_back(&param_info);
        }

        template <class DTYPE, class CTYPE, ct::Flag_t FLAGS, class METADATA, ct::index_t I>
        static void paramInfo(std::vector<mo::ParamInfo*>& vec,
                              ct::MemberObjectPointer<mo::TParamPtr<DTYPE> CTYPE::*, FLAGS, METADATA> ptr,
                              ct::Indexer<I>)
        {
            static mo::ParamInfo param_info(
                mo::TypeInfo(typeid(DTYPE)), ptr.m_name.slice(0, -6).toString(), "", "", mo::ParamFlags::kCONTROL, "");
            vec.push_back(&param_info);
        }

        static void paramInfoRecurse(std::vector<mo::ParamInfo*>& vec, ct::Indexer<0> idx)
        {
            auto ptr = ct::Reflect<T>::getPtr(idx);
            paramInfo(vec, ptr, idx);
        }

        template <ct::index_t I>
        static void paramInfoRecurse(std::vector<mo::ParamInfo*>& vec, ct::Indexer<I> idx)
        {
            auto ptr = ct::Reflect<T>::getPtr(idx);
            paramInfo(vec, ptr, idx);
            paramInfoRecurse(vec, --idx);
        }

        static void getParamInfoStatic(std::vector<mo::ParamInfo*>& vec)
        {
            paramInfoRecurse(vec, ct::Reflect<T>::end());
        }

        static std::vector<mo::ParamInfo*> getParamInfoStatic()
        {
            std::vector<mo::ParamInfo*> vec;
            getParamInfoStatic(vec);
            return vec;
        }

        template <class R, class... Args>
        inline void operator()(const mo::Signal<mo::TSignal<R(Args...)>>& data,
                               const mo::Name& name,
                               const int,
                               const bool,
                               int32_t* counter)
        {
            // TODO finish signal initialization
            this->addSignal(data.get(), name.get());
            *counter += 1;
        }

        template <class PTR, ct::index_t I>
        int initSignal(PTR, const ct::Indexer<I>, const bool)
        {
            return 0;
        }

        template <class DTYPE, class CTYPE, ct::Flag_t FLAGS, class METADATA, ct::index_t I>
        int initSignal(ct::MemberObjectPointer<mo::TSignal<DTYPE> CTYPE::*, FLAGS, METADATA> ptr,
                       const ct::Indexer<I>,
                       const bool)
        {
            this->addSignal(&ptr.set(*this), ptr.m_name);
            return 1;
        }

        void initSignal(const bool first_init, int32_t* counter, const ct::Indexer<0> idx)
        {
            *counter += initSignal(ct::Reflect<T>::getPtr(idx), idx, first_init);
        }

        template <index_t I>
        void initSignal(const bool first_init, int32_t* counter, const ct::Indexer<I> idx)
        {
            *counter += initSignal(ct::Reflect<T>::getPtr(idx), idx, first_init);
            initSignal(first_init, counter, --idx);
        }

        int initSignals(bool first_init) override
        {
            int32_t count = 0;
            initSignal(first_init, &count, ct::Reflect<T>::end());
            return count;
        }

        struct SignalInfoVisitor
        {
            template <ct::index_t N>
            inline void operator()(const mo::Type& type, const mo::Name& name, const ct::Indexer<N>)
            {
                static mo::SignalInfo info{type.get(), std::string(name.get()), "", ""};
                vec.push_back(&info);
            }

            void getSignalInfoParents(ct::VariadicTypedef<void>* = nullptr)
            {
            }

            template <class Parent>
            void getSignalInfoParents(ct::VariadicTypedef<Parent>* = nullptr)
            {
                Parent::template InterfaceHelper<Parent>::getSignalInfoStatic(vec);
            }

            template <class Parent, class... Parents>
            void getSignalInfoParents(ct::VariadicTypedef<Parent, Parents...>* = nullptr)
            {
                Parent::template InterfaceHelper<Parent>::getSignalInfoStatic(vec);
                getSignalInfoParents(static_cast<ct::VariadicTypedef<Parents...>*>(nullptr));
            }
            std::vector<mo::SignalInfo*>& vec;
        };

        template <class PTR, ct::index_t I>
        static void signalInfo(std::vector<mo::SignalInfo*>&, PTR, ct::Indexer<I>)
        {
            // ignore for now, added in the above initParam for TParamPtr
        }

        template <class SIG, class CTYPE, ct::Flag_t FLAGS, class METADATA, ct::index_t I>
        static void signalInfo(std::vector<mo::SignalInfo*>& vec,
                               ct::MemberObjectPointer<TSignal<SIG> CTYPE::*, FLAGS, METADATA> ptr,
                               ct::Indexer<I>)
        {
            // ignore for now, added in the above initParam for TParamPtr
            static SignalInfo info{mo::TypeInfo(typeid(SIG)), ptr.m_name, "", ""};
            vec.push_back(&info);
        }

        static void signalInfo(std::vector<mo::SignalInfo*>& vec, ct::Indexer<0> idx)
        {
            auto ptr = ct::Reflect<T>::getPtr(idx);
            signalInfo(vec, ptr, idx);
        }

        template <ct::index_t I>
        static void signalInfo(std::vector<mo::SignalInfo*>& vec, ct::Indexer<I> idx)
        {
            auto ptr = ct::Reflect<T>::getPtr(idx);
            signalInfo(vec, ptr, idx);
            signalInfo(vec, --idx);
        }

        static void getSignalInfoStatic(std::vector<mo::SignalInfo*>& vec)
        {
            signalInfo(vec, ct::Reflect<T>::end());
        }

        static std::vector<mo::SignalInfo*> getSignalInfoStatic()
        {
            std::vector<mo::SignalInfo*> vec;
            getSignalInfoStatic(vec);
            return vec;
        }

        template <class Sig, class R, class T1, class... Args>
        inline void operator()(const mo::Slot<Sig>& data,
                               const mo::NamedType<R (T1::*)(Args...), Function>& fptr,
                               const mo::Name& name,
                               int /*N*/,
                               bool /*first_init*/)
        {
            (*data.get()) =
                ct::variadicBind(fptr.get(), static_cast<T*>(this), ct::make_int_sequence<sizeof...(Args)>{});
            this->addSlot(data.get(), name.get());
        }

        template <index_t I, class U, ct::Flag_t FLAGS, class METADATA, class... PTRS>
        void bindSlot(Indexer<I>, ct::MemberFunctionPointers<U, FLAGS, METADATA, PTRS...> ptrs)
        {
            using BoundSig_t = typename std::decay<decltype(std::get<I>(ptrs.m_ptrs))>::type::BoundSig_t;
            std::unique_ptr<mo::ISlot> slot(new mo::TSlot<BoundSig_t>(ptrs.template bind<I>(this)));
            this->addSlot(std::move(slot), ptrs.m_name);
        }

        template <class U, ct::Flag_t FLAGS, class METADATA, class... PTRS>
        void bindSlotRecurse(Indexer<0> idx, ct::MemberFunctionPointers<U, FLAGS, METADATA, PTRS...> ptrs)
        {
            bindSlot(idx, ptrs);
        }

        template <ct::index_t I, class U, ct::Flag_t FLAGS, class METADATA, class... PTRS>
        void bindSlotRecurse(Indexer<I> idx, ct::MemberFunctionPointers<U, FLAGS, METADATA, PTRS...> ptrs)
        {
            bindSlot(idx, ptrs);
            bindSlotRecurse(--idx, ptrs);
        }

        template <class U, ct::Flag_t FLAGS, class METADATA, class... PTRS, ct::index_t I>
        void bindSlot(ct::MemberFunctionPointers<U, FLAGS, METADATA, PTRS...> ptrs, ct::Indexer<I>)
        {
            // register a slot here
            bindSlotRecurse(Indexer<sizeof...(PTRS) - 1>{}, ptrs);
        }

        template <class PTR, ct::index_t I>
        void bindSlot(PTR, ct::Indexer<I>)
        {
        }

        void bindSlotsRecurse(const bool, const ct::Indexer<0> idx)
        {
            const auto ptr = ct::Reflect<T>::getPtr(idx);
            bindSlot(ptr, idx);
        }

        template <ct::index_t I>
        void bindSlotsRecurse(const bool first_init, const ct::Indexer<I> idx)
        {
            const auto ptr = ct::Reflect<T>::getPtr(idx);
            bindSlot(ptr, idx);
            bindSlotsRecurse(first_init, --idx);
        }

        void bindSlots(bool first_init) override
        {
            bindSlotsRecurse(first_init, ct::Reflect<T>::end());
        }

        static std::vector<std::pair<mo::ISlot*, std::string>> getStaticSlots()
        {
            std::vector<std::pair<mo::ISlot*, std::string>> static_slots;

            return static_slots;
        }

        template <class U, ct::Flag_t FLAGS, class METADATA, class... PTRS, index_t I>
        static void slotInfo(std::vector<mo::SlotInfo*>& vec,
                             ct::MemberFunctionPointers<U, FLAGS, METADATA, PTRS...> ptrs,
                             ct::Indexer<I>,
                             ct::Indexer<0>)
        {
            using BoundSig_t = typename std::decay<decltype(std::get<0>(ptrs.m_ptrs))>::type::BoundSig_t;
            static SlotInfo info{TypeInfo(typeid(BoundSig_t)), ptrs.m_name, "", "", false};
            vec.push_back(&info);
        }

        template <class U, ct::Flag_t FLAGS, class METADATA, class... PTRS, index_t I, index_t J>
        static void slotInfo(std::vector<mo::SlotInfo*>& vec,
                             ct::MemberFunctionPointers<U, FLAGS, METADATA, PTRS...> ptrs,
                             ct::Indexer<I> idxi,
                             ct::Indexer<J> idxj)
        {
            using BoundSig_t = typename std::decay<decltype(std::get<J>(ptrs.m_ptrs))>::type::BoundSig_t;
            static SlotInfo info{TypeInfo(typeid(BoundSig_t)), ptrs.m_name, "", "", false};
            vec.push_back(&info);
            slotInfo(vec, ptrs, idxi, --idxj);
        }

        // Dispatch to all overloads
        template <class U, ct::Flag_t FLAGS, class METADATA, class... PTRS, ct::index_t I>
        static void slotInfo(std::vector<mo::SlotInfo*>& vec,
                             ct::MemberFunctionPointers<U, FLAGS, METADATA, PTRS...> ptrs,
                             ct::Indexer<I> idx)
        {
            slotInfo(vec, ptrs, idx, ct::Indexer<sizeof...(PTRS) - 1>{});
        }

        // Static functions
        template <class U, ct::Flag_t FLAGS, class METADATA, class... PTRS, ct::index_t I>
        static void
        slotInfo(std::vector<mo::SlotInfo*>&, ct::StaticFunctions<U, FLAGS, METADATA, PTRS...>, ct::Indexer<I>)
        {
            // slotInfo(vec, ptrs, idx, ct::Indexer<sizeof...(PTRS)-1>{});
        }

        // skip non slots
        template <class PTR, ct::index_t I>
        static void slotInfo(std::vector<mo::SlotInfo*>&, PTR, ct::Indexer<I>)
        {
        }

        static void slotInfo(std::vector<mo::SlotInfo*>& vec, ct::Indexer<0> idx)
        {
            auto ptr = ct::Reflect<T>::getPtr(idx);
            slotInfo(vec, ptr, idx);
        }

        template <index_t I>
        static void slotInfo(std::vector<mo::SlotInfo*>& vec, ct::Indexer<I> idx)
        {
            auto ptr = ct::Reflect<T>::getPtr(idx);
            slotInfo(vec, ptr, idx);
            slotInfo(vec, --idx);
        }

        static void getSlotInfoStatic(std::vector<mo::SlotInfo*>& vec)
        {
            slotInfo(vec, ct::Reflect<T>::end());
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

        static std::string tooltipStatic()
        {
            return _get_tooltip_helper<T>();
        }

        static std::string descriptionStatic()
        {
            return _get_description_helper<T>();
        }

        static TypeInfo getTypeInfoStatic()
        {
            return TypeInfo(typeid(typename T::BASE_CLASS));
        }

        std::vector<std::pair<ISlot*, std::string>> getStaticSlots() const override
        {
            return T::getStaticSlots();
        }

        std::vector<ParamInfo*> getParamInfo() const override
        {
            std::vector<ParamInfo*> info;
            getParamInfoStatic(info);
            return info;
        }

        std::vector<SignalInfo*> getSignalInfo() const override
        {
            std::vector<SignalInfo*> info;
            getSignalInfoStatic(info);
            return info;
        }

        std::vector<SlotInfo*> getSlotInfo() const override
        {
            std::vector<SlotInfo*> info;
            getSlotInfoStatic(info);
            return info;
        }

        std::string getObjectTooltip() const override
        {
            return tooltipStatic();
        }

        std::string getObjectHelp() const override
        {
            return descriptionStatic();
        }

        TypeInfo getTypeInfo() const override
        {
            return getTypeInfoStatic();
        }

        std::string GetObjectName() const override
        {
            return T::GetTypeNameStatic();
        }

        unsigned int GetInterfaceId() const override
        {
            return T::getHash();
        }

        std::string GetInterfaceName() const override
        {
            return T::GetInterfaceName();
        }

        IObjectConstructor* GetConstructor() const override
        {
            return T::GetConstructorStatic();
        }

        bool InheritsFrom(InterfaceID iid) const override
        {
            return T::InheritsFrom(iid);
        }

      private:
        DEFINE_HAS_STATIC_FUNCTION(HasTooltip, getTooltipStatic, std::string);
        DEFINE_HAS_STATIC_FUNCTION(HasDescription, getDescriptionStatic, std::string);
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
} // namespace mo
#endif
