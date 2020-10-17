#ifndef MO_VISITATION_STRUCT_TRAITS_HPP
#define MO_VISITATION_STRUCT_TRAITS_HPP
#include "IDynamicVisitor.hpp"
#include "TraitInterface.hpp"
#include "TraitRegistry.hpp"

#include <ct/reflect.hpp>
#include <ct/reflect/visitor.hpp>

namespace mo
{
    template <class T>
    struct StructBase : virtual IStructTraits
    {
        using base = IStructTraits;
        static constexpr const bool DEFAULT = false;
        StructBase()
        {
        }

        size_t size() const override
        {
            return sizeof(T);
        }

        bool triviallySerializable() const override
        {
            return std::is_trivially_copyable<T>::value;
        }

        TypeInfo type() const override
        {
            return TypeInfo::create<T>();
        }

        T* ptr(void* inst) const
        {
            return static_cast<T*>(inst);
        }

        const T* ptr(const void* inst) const
        {
            return static_cast<const T*>(inst);
        }

        T& ref(void* inst) const
        {
            return *ptr(inst);
        }

        const T& ref(const void* inst) const
        {
            return *ptr(inst);
        }
    };

    struct RuntimeVisitorParams : ct::DefaultVisitorParams
    {
        constexpr static const bool ACCUMULATE_PATH = false;

        template <class T>
        constexpr static bool visitMemberFunctions(T)
        {
            return false;
        }
        template <class T>
        constexpr static bool visitMemberFunction(T)
        {
            return false;
        }
        template <class T>
        constexpr static bool visitStaticFunctions(T)
        {
            return false;
        }
        template <class T>
        constexpr static bool visitStaticFunction(T)
        {
            return false;
        }
    };

    using index_t = ct::index_t;
    template <index_t N>
    using Indexer = ct::Indexer<N>;

    template <class T, index_t I>
    auto visitValue(ILoadVisitor& visitor, T& obj, const Indexer<I> idx) -> ct::EnableIf<ct::IsWritable<T, I>::value>
    {
        auto accessor = ct::Reflect<T>::getPtr(idx);
        using Ret_t = decltype(accessor.set(obj));
        Ret_t tmp = accessor.set(obj);
        auto ptr = &tmp;
        const auto name = ct::getName<I, T>();
        visitor(ptr, name);
    }

    template <class T, index_t I>
    auto visitValue(ILoadVisitor&, T&, const Indexer<I>) -> ct::EnableIf<!ct::IsWritable<T, I>::value>
    {
    }

    template <class T>
    void visitHelper(ILoadVisitor& visitor, T& obj, const Indexer<0> idx)
    {
        visitValue(visitor, obj, idx);
    }

    template <class T, index_t I>
    void visitHelper(ILoadVisitor& visitor, T& obj, const Indexer<I> idx)
    {
        const auto next_index = --idx;
        visitHelper(visitor, obj, next_index);
        visitValue(visitor, obj, idx);
    }

    template <class T, index_t I>
    auto visitValue(ISaveVisitor& visitor, const T& obj, const ct::Indexer<I> idx)
        -> ct::EnableIf<ct::IsWritable<T, I>::value>
    {
        auto accessor = ct::Reflect<T>::getPtr(idx);
        using RefType = typename ct::ReferenceType<typename ct::GetType<decltype(accessor)>::type>::ConstType;
        RefType ref = static_cast<RefType>(accessor.get(obj));
        auto name = ct::getName<I, T>();
        visitor(&ref, name);
    }

    template <class T, index_t I>
    auto visitValue(ISaveVisitor&, const T&, const ct::Indexer<I>) -> ct::EnableIf<!ct::IsWritable<T, I>::value>
    {
    }

    template <class T>
    void visitHelper(ISaveVisitor& visitor, const T& obj, const Indexer<0> idx)
    {
        visitValue(visitor, obj, idx);
    }

    template <class T, index_t I>
    void visitHelper(ISaveVisitor& visitor, const T& obj, const Indexer<I> idx)
    {
        auto next_index = --idx;
        visitHelper(visitor, obj, next_index);
        visitValue(visitor, obj, idx);
    }

    template <class T, index_t I, class U = void>
    using EnableVisitation =
        ct::EnableIf<!ct::IsMemberFunction<T, I>::value &&
                         !ct::IsEnumField<decltype(ct::Reflect<T>::getPtr(ct::Indexer<I>{}))>::value,
                     U>;

    template <class T, index_t I, class U = void>
    using DisableVisitation =
        ct::EnableIf<ct::IsMemberFunction<T, I>::value ||
                         ct::IsEnumField<decltype(ct::Reflect<T>::getPtr(ct::Indexer<I>{}))>::value,
                     U>;

    template <class T, index_t I>
    auto visitValue(StaticVisitor& visitor, const Indexer<I> idx) -> EnableVisitation<T, I>
    {
        using Type = typename ct::GetType<decltype(ct::Reflect<T>::getPtr(idx))>::type;
        const auto name = ct::getName<I, T>();
        visitor.visit<typename std::decay<Type>::type>(name);
    }

    template <class T, index_t I>
    auto visitValue(StaticVisitor&, const Indexer<I>) -> DisableVisitation<T, I>
    {
    }

    template <class T>
    void visitHelper(StaticVisitor& visitor, const Indexer<0> idx)
    {
        visitValue<T>(visitor, idx);
    }

    template <class T, index_t I>
    void visitHelper(StaticVisitor& visitor, const Indexer<I> idx)
    {
        const auto next_idx = --idx;
        visitHelper<T>(visitor, next_idx);
        visitValue<T>(visitor, idx);
    }

    template <class T, ct::index_t I>
    ct::EnableIf<ct::IsMemberObject<T, I>::value, bool> getMemberHelper(
        T& inst, void** member, const IStructTraits** trait, ct::Indexer<I> itr, std::string* name = nullptr)
    {
        auto ptr = ct::Reflect<T>::getPtr(itr);
        using DType = ct::decay_t<typename decltype(ptr)::Data_t>;
        static const TTraits<DType> member_trait;
        *trait = &member_trait;
        *member = static_cast<void*>(&ptr.set(inst));
        if (name)
        {
            *name = ptr.getName();
        }
        return true;
    }

    template <class T, ct::index_t I>
    ct::EnableIf<!ct::IsMemberObject<T, I>::value, bool> getMemberHelper(
        T& inst, void** member, const IStructTraits** trait, ct::Indexer<I> itr, std::string* name = nullptr)
    {

        return false;
    }

    template <class T>
    bool getMemberRecurse(T& inst,
                          void** member,
                          const IStructTraits** trait,
                          uint32_t idx,
                          ct::Indexer<0> itr,
                          std::string* name = nullptr)
    {
        if (idx == itr)
        {
            getMemberHelper(inst, member, trait, itr, name);
            return true;
        }
        else
        {
            return false;
        }
    }

    template <class T, ct::index_t I>
    bool getMemberRecurse(T& inst,
                          void** member,
                          const IStructTraits** trait,
                          uint32_t idx,
                          ct::Indexer<I> itr,
                          std::string* name = nullptr)
    {
        if (idx == itr)
        {
            getMemberHelper(inst, member, trait, itr, name);
            return true;
        }
        else
        {
            const ct::Indexer<I - 1> next_itr;
            return getMemberRecurse(inst, member, trait, idx, next_itr, name);
        }
    }

    // const versions
    template <class T, ct::index_t I>
    ct::EnableIf<ct::IsMemberObject<T, I>::value, bool> getMemberHelper(const T& inst,
                                                                        const void** member,
                                                                        const IStructTraits** trait,
                                                                        ct::Indexer<I> itr,
                                                                        std::string* name = nullptr)
    {
        auto ptr = ct::Reflect<T>::getPtr(itr);
        using DType = ct::decay_t<typename decltype(ptr)::Data_t>;
        static const TTraits<DType> member_trait;
        *trait = &member_trait;
        *member = static_cast<const void*>(&ptr.get(inst));
        if (name)
        {
            *name = ptr.getName();
        }
        return true;
    }

    template <class T, ct::index_t I>
    ct::EnableIf<!ct::IsMemberObject<T, I>::value, bool> getMemberHelper(const T& inst,
                                                                         const void** member,
                                                                         const IStructTraits** trait,
                                                                         ct::Indexer<I> itr,
                                                                         std::string* name = nullptr)
    {

        return false;
    }

    template <class T>
    bool getMemberRecurse(const T& inst,
                          const void** member,
                          const IStructTraits** trait,
                          uint32_t idx,
                          ct::Indexer<0> itr,
                          std::string* name = nullptr)
    {
        if (idx == itr)
        {
            return getMemberHelper(inst, member, trait, itr, name);
        }
        else
        {
            return false;
        }
    }

    template <class T, ct::index_t I>
    bool getMemberRecurse(const T& inst,
                          const void** member,
                          const IStructTraits** trait,
                          uint32_t idx,
                          ct::Indexer<I> itr,
                          std::string* name = nullptr)
    {
        if (idx == itr)
        {
            return getMemberHelper(inst, member, trait, itr, name);
        }
        else
        {
            const ct::Indexer<I - 1> next_itr;
            return getMemberRecurse(inst, member, trait, idx, next_itr, name);
        }
    }

    template <class T>
    struct TTraits<T, 4, ct::EnableIfReflected<T>> : virtual StructBase<T>
    {
        using MemberObjectTypes = typename ct::GlobMemberObjects<T>::types;
        using WritableObjectTypes = typename ct::GlobWritable<T>::types;
        static constexpr const uint32_t num_member_objects = ct::GlobMemberObjects<T>::num + ct::GlobWritable<T>::num;

        void load(ILoadVisitor& visitor, void* inst, const std::string& name, size_t) const override
        {
            auto ptr = static_cast<T*>(inst);
            const auto idx = ct::Reflect<T>::end();
            visitHelper(visitor, *ptr, idx);
        }

        void save(ISaveVisitor& visitor, const void* inst, const std::string& name, size_t) const override
        {
            auto ptr = static_cast<const T*>(inst);
            const auto idx = ct::Reflect<T>::end();
            visitHelper(visitor, *ptr, idx);
        }

        void visit(StaticVisitor& visitor, const std::string& name) const override
        {
            // TODO
            const auto idx = ct::Reflect<T>::end();
            visitHelper<T>(visitor, idx);
        }

        std::string name() const override
        {
            return ct::Reflect<T>::getTypeName();
        }

        uint32_t getNumMembers() const override
        {
            return num_member_objects;
        }

        bool getMember(
            void* inst, void** member, const IStructTraits** trait, uint32_t idx, std::string* name) const override
        {
            auto& ref = this->ref(inst);
            const auto itr = ct::Reflect<T>::end();
            return getMemberRecurse(ref, member, trait, idx, itr, name);
        }

        bool getMember(const void* inst,
                       const void** member,
                       const IStructTraits** trait,
                       uint32_t idx,
                       std::string* name) const override
        {
            auto& ref = this->ref(inst);
            const auto itr = ct::Reflect<T>::end();
            return getMemberRecurse(ref, member, trait, idx, itr, name);
        }
    };

    template <class T>
    struct TTraits<T, 9, ct::EnableIf<IsPrimitiveRuntimeReflected<T>::value>> : virtual StructBase<T>
    {
        void load(ILoadVisitor& visitor, void* inst, const std::string& name, size_t) const override
        {
            T* ptr = static_cast<T*>(inst);
            visitor(ptr, name);
        }

        void save(ISaveVisitor& visitor, const void* inst, const std::string& name, size_t) const override
        {
            const T* ptr = static_cast<const T*>(inst);
            visitor(ptr, name);
        }

        void visit(StaticVisitor& visitor, const std::string& name) const override
        {
            visitor.template visit<T>("value");
        }

        uint32_t getNumMembers() const override
        {
            return 0;
        }

        bool getMember(
            void* inst, void** member, const IStructTraits** trait, uint32_t idx, std::string* name) const override
        {
            return false;
        }

        bool getMember(const void* inst,
                       const void** member,
                       const IStructTraits** trait,
                       uint32_t idx,
                       std::string* name) const override
        {
            return false;
        }
    };
} // namespace mo

#endif // MO_VISITATION_STRUCT_TRAITS_HPP
