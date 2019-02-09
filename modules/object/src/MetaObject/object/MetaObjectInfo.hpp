#ifndef MO_OBJECT_META_OBJECT_INFO_HPP
#define MO_OBJECT_META_OBJECT_INFO_HPP
#include "IMetaObjectInfo.hpp"

#include "MetaObject/core/detail/HelperMacros.hpp"
#include "MetaObject/core/detail/Placeholders.hpp"

#include "MetaObject/object/MetaObjectInfoDatabase.hpp"
#include "MetaObject/object/detail/MetaObjectMacros.hpp"

#include "MetaObject/params/ParamInfo.hpp"
#include "MetaObject/params/TInputParam.hpp"
#include "MetaObject/params/TParamPtr.hpp"

#include "MetaObject/signals/SignalInfo.hpp"
#include "MetaObject/signals/SlotInfo.hpp"

#include <RuntimeObjectSystem/ISimpleSerializer.h>

#include "ct/VariadicTypedef.hpp"
#include <ct/Indexer.hpp>

#include <type_traits>

struct ISimpleSerializer;

namespace mo
{
    template <class... T>
    class TMultiInput;
    template <class... T>
    class TMultiOutput;

    template <class T>
    struct TMetaObjectInterfaceHelper : public T
    {
        template <class DType, class BufferFlags>
        inline void operator()(const mo::Data<DType>&,
                               const mo::Name& name,
                               const mo::Param<BufferFlags>& param,
                               const int32_t,
                               const bool)
        {
            param.get()->setName(name.get());
            T::addParam(param.get());
        }

        template <class DType>
        inline void operator()(const mo::Data<DType>& data,
                               const mo::Name& name,
                               const mo::Param<mo::TParamPtr<DType>>& param,
                               const int32_t,
                               const bool)
        {
            param.get()->setName(name.get());
            param.get()->updatePtr(data.get());
            T::addParam(param.get());
        }

        template <class DType>
        inline void operator()(const mo::Data<DType>& data,
                               const mo::Name& name,
                               const mo::Param<mo::TParamOutput<DType>>& param,
                               const int32_t,
                               const bool)
        {
            param.get()->setName(name.get());
            param.get()->updatePtr(data.get());
            T::addParam(param.get());
        }

        template <class DType>
        inline void operator()(const mo::Data<DType const*>& data,
                               const mo::Name& name,
                               const mo::Param<mo::TInputParamPtr<DType>>& param,
                               const int32_t,
                               const bool)
        {
            param.get()->setName(name.get());
            param.get()->setUserDataPtr(data.get());
            T::addParam(param.get());
        }

        template <class... DTypes>
        inline void operator()(const mo::Data<std::tuple<const DTypes*...>>& data,
                               const mo::Name& name,
                               const mo::Param<mo::TMultiInput<DTypes...>>& param,
                               const int32_t,
                               const bool)
        {
            param.get()->setName(name.get());
            param.get()->setUserDataPtr(data.get());
            T::addParam(param.get());
        }

        template <class... DTypes>
        inline void
        operator()(const mo::Name& name, const mo::Param<mo::TMultiOutput<DTypes...>>& param, const int32_t, const bool)
        {
            param.get()->setName(name.get());
            T::addParam(param.get());
        }

        void initParamsRecurse(std::map<std::string, std::pair<void*, mo::TypeInfo>>& params,
                               const bool first_init,
                               const ct::Indexer<0> idx)
        {
            const auto ptr = ct::Reflect<T>::getPtr(idx);
        }

        template <ct::index_t I>
        void initParamsRecurse(std::map<std::string, std::pair<void*, mo::TypeInfo>>& params,
                               const bool first_init,
                               const ct::Indexer<I> idx)
        {
            // const auto ptr = ct::Reflect<T>::getPtr(idx);

            initParamsRecurse(params, first_init, --idx);
        }

        void initParams(bool first_init) override
        {
            std::map<std::string, std::pair<void*, mo::TypeInfo>> params;
            initParamsRecurse(params, first_init, ct::Reflect<T>::end());
        }

        struct LoadVisitorHelper
        {
            template <class DType, class ParamType, ct::index_t N>
            inline void operator()(mo::Data<DType> data, mo::Name name, ParamType, const ct::Indexer<N>)
            {
                visitor(data.get(), name.get());
            }

            ILoadVisitor& visitor;
        };

        void load(ILoadVisitor& visitor)
        {
            LoadVisitorHelper vis{visitor};
        }

        struct SaveVisitorHelper
        {
            template <class DType, class ParamType, ct::index_t N>
            inline void operator()(const mo::Data<DType>& data, const mo::Name& name, ParamType, const ct::Indexer<N>)
            {
                visitor(data.get(), name.get());
            }

            ISaveVisitor& visitor;
        };

        void save(ISaveVisitor& visitor) override
        {
            SaveVisitorHelper vis{visitor};
        }

        struct RCCSerializationVisitor
        {
            ISimpleSerializer* serializer;
            template <class DType, class ParamType>
            inline void operator()(const mo::Data<DType>& data, const mo::Name& name, ParamType, ct::index_t)
            {
                serializer->SerializeProperty(name.get(), *data.get());
            }
        };

        void serializeParams(ISimpleSerializer* serializer) override
        {
            RCCSerializationVisitor visitor{serializer};
        }

        struct ParamInfoVisitor
        {
            template <ct::index_t N>
            inline void operator()(const mo::Name& name, const Type& type, ct::Indexer<N>)
            {
                static mo::ParamInfo info(type.get(), name.get(), "", "", flags);
                vec.push_back(&info);
            }

            void getParamInfoParents(ct::VariadicTypedef<void>* = nullptr)
            {
            }

            template <class Parent>
            void getParamInfoParents(ct::VariadicTypedef<Parent>* = nullptr)
            {
                Parent::template InterfaceHelper<Parent>::getParamInfoStatic(vec);
            }

            template <class Parent, class... Parents>
            void getParamInfoParents(ct::VariadicTypedef<Parent, Parents...>* = nullptr)
            {
                Parent::template InterfaceHelper<Parent>::getParamInfoStatic(vec);
                getParamInfoParents(static_cast<ct::VariadicTypedef<Parents...>*>(nullptr));
            }
            std::vector<mo::ParamInfo*>& vec;
            mo::ParamFlags flags;
        };

        static void getParamInfoStatic(std::vector<mo::ParamInfo*>& vec)
        {
            {
                ParamInfoVisitor visitor{vec, mo::ParamFlags::Control_e};
            }
            {
                ParamInfoVisitor visitor{vec, mo::ParamFlags::Input_e};
            }

            {
                ParamInfoVisitor visitor{vec, mo::ParamFlags::Output_e};
            }
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

        int initSignals(bool first_init) override
        {
            int32_t count = 0;
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

        static void getSignalInfoStatic(std::vector<mo::SignalInfo*>& vec)
        {
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
            (*data.get()) = variadicBind(fptr.get(), static_cast<T*>(this), make_int_sequence<sizeof...(Args)>{});
            this->addSlot(data.get(), name.get());
        }

        void bindSlots(bool first_init) override
        {
        }

        struct SlotInfoVisitor
        {
            template <class R, class T1, class... Args, ct::index_t N>
            inline void operator()(const mo::NamedType<R (T1::*)(Args...), Function>& /*fptr*/,
                                   const mo::Name& name,
                                   ct::Indexer<N>)
            {
                static mo::SlotInfo info{mo::TypeInfo(typeid(R(Args...))), std::string(name.get()), "", "", false};
                vec.push_back(&info);
            }

            template <class R, class... Args, ct::index_t N>
            inline void operator()(const mo::NamedType<R (*)(Args...), StaticFunction>& /*fptr*/,
                                   const mo::Name& name,
                                   ct::Indexer<N>)
            {
                static mo::SlotInfo info{mo::TypeInfo(typeid(R(Args...))), std::string(name.get()), "", "", true};
                vec.push_back(&info);
            }

            void getSlotInfoParents(ct::VariadicTypedef<void>* = nullptr)
            {
            }

            template <class Parent>
            void getSlotInfoParents(ct::VariadicTypedef<Parent>* = nullptr)
            {
                Parent::template InterfaceHelper<Parent>::getSlotInfoStatic(vec);
            }

            template <class Parent, class... Parents>
            void getSlotInfoParents(ct::VariadicTypedef<Parent, Parents...>* = nullptr)
            {
                Parent::template InterfaceHelper<Parent>::getSlotInfoStatic(vec);
                getSlotInfoParents(static_cast<ct::VariadicTypedef<Parents...>*>(nullptr));
            }

            std::vector<mo::SlotInfo*>& vec;
        };

        struct StaticSlotVisitor
        {
            std::vector<std::pair<mo::ISlot*, std::string>>& slot_vec;

            template <class R, class... Args, ct::index_t N>
            inline void operator()(const mo::NamedType<R (*)(Args...), StaticFunction>& fptr,
                                   const mo::Name& name,
                                   const ct::Indexer<N>)
            {
                static mo::TSlot<R(Args...)> slot = variadicBind(fptr.get(), make_int_sequence<sizeof...(Args)>{});
                slot_vec.push_back(std::make_pair(&slot, name.get()));
            }

            template <class R, class T1, class... Args, ct::index_t N>
            inline void operator()(const mo::NamedType<R (T1::*)(Args...), Function>&, const mo::Name&, ct::Indexer<N>)
            {
            }
        };

        static std::vector<std::pair<mo::ISlot*, std::string>> getStaticSlots()
        {
            std::vector<std::pair<mo::ISlot*, std::string>> static_slots;

            return static_slots;
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

        std::string getObjectTooltip() const
        {
            return tooltipStatic();
        }

        std::string getObjectHelp() const
        {
            return descriptionStatic();
        }

        TypeInfo getTypeInfo() const
        {
            return getTypeInfoStatic();
        }

        std::string GetObjectName() const
        {
            return T::GetTypeNameStatic();
        }

        unsigned int GetInterfaceId() const
        {
            return T::getHash();
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
#endif
