#ifndef MO_VISITATION_STRUCT_TRAITS_HPP
#define MO_VISITATION_STRUCT_TRAITS_HPP
#include "export.hpp"

#include "TraitInterface.hpp"
#include "TraitRegistry.hpp"

#include "type_traits.hpp"

#include <ct/reflect.hpp>
#include <ct/reflect/visitor.hpp>
#include <ct/reflect_traits.hpp>

namespace mo
{
    template <class T>
    struct StructBase : virtual IStructTraits
    {
        using base = IStructTraits;
        static constexpr const bool DEFAULT = false;
        StructBase();

        size_t size() const override;

        bool triviallySerializable() const override;

        TypeInfo type() const override;

        T* ptr(void* inst) const;

        const T* ptr(const void* inst) const;

        T& ref(void* inst) const;

        const T& ref(const void* inst) const;
    };

    template <class T>
    struct PtrBase : virtual IPtrTraits
    {
        using base = IStructTraits;
        static constexpr const bool DEFAULT = false;

        PtrBase();

        size_t size() const override;

        bool triviallySerializable() const override;

        TypeInfo type() const override;

        T* ptr(void* inst) const;

        const T* ptr(const void* inst) const;

        T& ref(void* inst) const;

        const T& ref(const void* inst) const;
    };

    struct RuntimeVisitorParams : ct::DefaultVisitorParams
    {
        constexpr static const bool ACCUMULATE_PATH = false;

        template <class T>
        constexpr static bool visitMemberFunctions(T);

        template <class T>
        constexpr static bool visitMemberFunction(T);

        template <class T>
        constexpr static bool visitStaticFunctions(T);

        template <class T>
        constexpr static bool visitStaticFunction(T);
    };

    using index_t = ct::index_t;
    template <index_t N>
    using Indexer = ct::Indexer<N>;

    template <class T>
    struct TTraits<T, 4, ct::EnableIfReflected<T>> : virtual StructBase<T>
    {
        using MemberObjectTypes = typename ct::GlobMemberObjects<T>::types;
        using WritableObjectTypes = typename ct::GlobWritable<T>::types;
        static constexpr const uint32_t num_member_objects = ct::GlobMemberObjects<T>::num + ct::GlobWritable<T>::num;

        void load(ILoadVisitor& visitor, void* inst, const std::string&, size_t) const override;

        void save(ISaveVisitor& visitor, const void* inst, const std::string&, size_t) const override;

        void visit(StaticVisitor& visitor, const std::string&) const override;

        std::string name() const override;

        uint32_t getNumMembers() const override;

        bool loadMember(ILoadVisitor& visitor, void* inst, uint32_t idx, std::string* name = nullptr) const override;

        bool
        saveMember(ISaveVisitor& visitor, const void* inst, uint32_t idx, std::string* name = nullptr) const override;
    };

    template <class T>
    struct TTraits<T, 9, ct::EnableIf<IsPrimitiveRuntimeReflected<T>::value>> : virtual StructBase<T>
    {
        void load(ILoadVisitor& visitor, void* inst, const std::string& name, size_t) const override;

        void save(ISaveVisitor& visitor, const void* inst, const std::string& name, size_t) const override;

        void visit(StaticVisitor& visitor, const std::string&) const override;

        uint32_t getNumMembers() const override;
    };
} // namespace mo

// implementation

#include "IDynamicVisitor.hpp"
namespace mo
{
    template <class T>
    StructBase<T>::StructBase()
    {
    }

    template <class T>
    size_t StructBase<T>::size() const
    {
        return sizeof(T);
    }

    template <class T>
    bool StructBase<T>::triviallySerializable() const
    {
        return std::is_trivially_copyable<T>::value;
    }

    template <class T>
    TypeInfo StructBase<T>::type() const
    {
        return TypeInfo::create<T>();
    }

    template <class T>
    T* StructBase<T>::ptr(void* inst) const
    {
        return static_cast<T*>(inst);
    }

    template <class T>
    const T* StructBase<T>::ptr(const void* inst) const
    {
        return static_cast<const T*>(inst);
    }

    template <class T>
    T& StructBase<T>::ref(void* inst) const
    {
        return *ptr(inst);
    }

    template <class T>
    const T& StructBase<T>::ref(const void* inst) const
    {
        return *ptr(inst);
    }

    template <class T>
    PtrBase<T>::PtrBase()
    {
    }

    template <class T>
    size_t PtrBase<T>::size() const
    {
        return sizeof(T);
    }

    template <class T>
    bool PtrBase<T>::triviallySerializable() const
    {
        return std::is_trivially_copyable<T>::value;
    }

    template <class T>
    TypeInfo PtrBase<T>::type() const
    {
        return TypeInfo::create<T>();
    }

    template <class T>
    T* PtrBase<T>::ptr(void* inst) const
    {
        return static_cast<T*>(inst);
    }

    template <class T>
    const T* PtrBase<T>::ptr(const void* inst) const
    {
        return static_cast<const T*>(inst);
    }

    template <class T>
    T& PtrBase<T>::ref(void* inst) const
    {
        return *ptr(inst);
    }

    template <class T>
    const T& PtrBase<T>::ref(const void* inst) const
    {
        return *ptr(inst);
    }

    template <class T>
    constexpr bool RuntimeVisitorParams::visitMemberFunctions(T)
    {
        return false;
    }

    template <class T>
    constexpr bool RuntimeVisitorParams::visitMemberFunction(T)
    {
        return false;
    }

    template <class T>
    constexpr bool RuntimeVisitorParams::visitStaticFunctions(T)
    {
        return false;
    }

    template <class T>
    constexpr bool RuntimeVisitorParams::visitStaticFunction(T)
    {
        return false;
    }

    template <class T, index_t I>
    auto visitValue(ILoadVisitor& visitor, T& obj, const Indexer<I> idx)
        -> ct::EnableIf<ct::IsWritable<T, I>::value, bool>
    {
        auto accessor = ct::Reflect<T>::getPtr(idx);
        using Ret_t = decltype(accessor.set(obj));
        Ret_t tmp = accessor.set(obj);
        auto ptr = &tmp;
        const auto name = ct::getName<I, T>();
        visitor(ptr, name);
        return true;
    }

    template <class T, index_t I>
    auto visitValue(ILoadVisitor&, T&, const Indexer<I>) -> ct::EnableIf<!ct::IsWritable<T, I>::value, bool>
    {
        return false;
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
        -> ct::EnableIf<ct::IsWritable<T, I>::value, bool>
    {
        auto accessor = ct::Reflect<T>::getPtr(idx);
        using RefType = typename ct::ReferenceType<typename ct::GetType<decltype(accessor)>::type>::ConstType;
        const std::string name = ct::getName<I, T>();
        RefType ref = static_cast<RefType>(accessor.get(obj));
        visitor(&ref, name);
        return true;
    }

    template <class T, index_t I>
    auto visitValue(ISaveVisitor&, const T&, const ct::Indexer<I>) -> ct::EnableIf<!ct::IsWritable<T, I>::value, bool>
    {
        return true;
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
        visitor.template visit<typename std::decay<Type>::type>(name);
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
    ct::EnableIf<ct::IsWritable<T, I>::value, bool>
    loadMemberHelper(ILoadVisitor& visitor, T& inst, ct::Indexer<I> itr, std::string* name = nullptr)
    {
        if (visitValue(visitor, inst, itr))
        {
            if (name)
            {
                auto ptr = ct::Reflect<T>::getPtr(itr);
                *name = ptr.getName();
            }
            return true;
        }

        return false;
    }

    template <class T, ct::index_t I>
    ct::EnableIf<!ct::IsWritable<T, I>::value, bool>
    loadMemberHelper(ILoadVisitor&, T&, ct::Indexer<I>, std::string* = nullptr)
    {
        return false;
    }

    template <class T>
    bool loadMemberRecurse(ILoadVisitor& visitor,
                           T& inst,
                           uint32_t idx,
                           ct::Indexer<0> itr,
                           uint32_t& member_counter,
                           std::string* name = nullptr)
    {
        bool success = false;
        if (idx == member_counter)
        {
            success = loadMemberHelper(visitor, inst, itr, name);
        }
        if (!success)
        {
            if (ct::IsWritable<T, 0>::value)
            {
                ++member_counter;
            }
            return false;
        }
        return true;
    }

    template <class T, ct::index_t I>
    bool loadMemberRecurse(ILoadVisitor& visitor,
                           T& inst,
                           uint32_t idx,
                           ct::Indexer<I> itr,
                           uint32_t& member_counter,
                           std::string* name = nullptr)
    {
        const ct::Indexer<I - 1> next_itr;
        bool success = loadMemberRecurse(visitor, inst, idx, next_itr, member_counter, name);
        if (!success)
        {
            if (member_counter == idx)
            {
                success = loadMemberHelper(visitor, inst, itr, name);
            }
        }
        if (!success)
        {
            if (ct::IsWritable<T, I>::value)
            {
                ++member_counter;
            }
            return false;
        }
        return true;
    }

    // const versions
    template <class T, ct::index_t I>
    ct::EnableIf<ct::IsWritable<T, I>::value, bool>
    saveMemberHelper(ISaveVisitor& visitor, const T& inst, ct::Indexer<I> itr, std::string* name = nullptr)
    {
        if (visitValue(visitor, inst, itr))
        {
            if (name)
            {
                auto ptr = ct::Reflect<T>::getPtr(itr);
                *name = ptr.getName();
            }
            return true;
        }

        return false;
    }

    template <class T, ct::index_t I>
    ct::EnableIf<!ct::IsWritable<T, I>::value, bool>
    saveMemberHelper(ISaveVisitor&, const T&, ct::Indexer<I>, std::string* = nullptr)
    {

        return false;
    }

    // Save by index
    template <class T>
    bool saveMemberRecurse(ISaveVisitor& visitor,
                           const T& inst,
                           uint32_t idx,
                           ct::Indexer<0> itr,
                           uint32_t& member_counter,
                           std::string* name = nullptr)
    {
        bool success = false;
        // desired index matches member counter, see if we can save this member
        if (idx == member_counter)
        {
            success = saveMemberHelper(visitor, inst, itr, name);
        }
        // we couldn't save this member, the true desired member must be later?
        if (!success)
        {
            if (ct::IsWritable<T, 0>::value)
            {
                ++member_counter;
            }
            return false;
        }
        return true;
    }

    template <class T, ct::index_t I>
    bool saveMemberRecurse(ISaveVisitor& visitor,
                           const T& inst,
                           uint32_t idx,
                           ct::Indexer<I> itr,
                           uint32_t& member_counter,
                           std::string* name = nullptr)
    {
        const ct::Indexer<I - 1> next_itr;
        bool success = saveMemberRecurse(visitor, inst, idx, next_itr, member_counter, name);
        // We're now walking forward through the members
        if (!success)
        {
            if (member_counter == idx)
            {
                success = saveMemberHelper(visitor, inst, itr, name);
            }
        }
        if (!success)
        {
            if (ct::IsWritable<T, I>::value)
            {
                ++member_counter;
            }
            return false;
        }
        return true;
    }

    // Save by name
    template <class T>
    bool saveMemberRecurse(
        ISaveVisitor& visitor, const T& inst, const std::string& name, ct::Indexer<0> itr, uint32_t* = nullptr)
    {
        auto ptr = ct::Reflect<T>::getPtr(itr);
        if (ptr.getName() == name)
        {
            return visitValue(visitor, inst, itr);
        }
        else
        {
            return false;
        }
    }

    template <class T, ct::index_t I>
    bool saveMemberRecurse(
        ISaveVisitor& visitor, const T& inst, const std::string& name, ct::Indexer<I> itr, uint32_t* idx = nullptr)
    {
        auto ptr = ct::Reflect<T>::getPtr(itr);
        if (ptr.getName() == name)
        {
            return visitValue(visitor, inst, itr);
        }
        else
        {
            const ct::Indexer<I - 1> next_itr;
            return saveMemberRecurse(visitor, inst, name, next_itr, idx);
        }
    }

    template <class T>
    void
    TTraits<T, 4, ct::EnableIfReflected<T>>::load(ILoadVisitor& visitor, void* inst, const std::string&, size_t) const
    {
        auto ptr = static_cast<T*>(inst);
        const auto idx = ct::Reflect<T>::end();
        visitHelper(visitor, *ptr, idx);
    }

    template <class T>
    void TTraits<T, 4, ct::EnableIfReflected<T>>::save(ISaveVisitor& visitor,
                                                       const void* inst,
                                                       const std::string&,
                                                       size_t) const
    {
        auto ptr = static_cast<const T*>(inst);
        const auto idx = ct::Reflect<T>::end();
        visitHelper(visitor, *ptr, idx);
    }

    template <class T>
    void TTraits<T, 4, ct::EnableIfReflected<T>>::visit(StaticVisitor& visitor, const std::string&) const
    {
        // TODO
        const auto idx = ct::Reflect<T>::end();
        visitHelper<T>(visitor, idx);
    }

    template <class T>
    std::string TTraits<T, 4, ct::EnableIfReflected<T>>::name() const
    {
        return ct::Reflect<T>::getTypeName();
    }

    template <class T>
    uint32_t TTraits<T, 4, ct::EnableIfReflected<T>>::getNumMembers() const
    {
        return num_member_objects;
    }

    template <class T>
    bool TTraits<T, 4, ct::EnableIfReflected<T>>::loadMember(ILoadVisitor& visitor,
                                                             void* inst,
                                                             uint32_t idx,
                                                             std::string* name) const
    {
        T& ref = this->ref(inst);
        const auto itr = ct::Reflect<T>::end();
        uint32_t member_counter = 0;
        return loadMemberRecurse(visitor, ref, idx, itr, member_counter, name);
    }

    template <class T>
    bool TTraits<T, 4, ct::EnableIfReflected<T>>::saveMember(ISaveVisitor& visitor,
                                                             const void* inst,
                                                             uint32_t idx,
                                                             std::string* name) const
    {
        const T& ref = this->ref(inst);
        const auto itr = ct::Reflect<T>::end();
        uint32_t member_counter = 0;
        return saveMemberRecurse(visitor, ref, idx, itr, member_counter, name);
    }

    template <class T>
    void TTraits<T, 9, ct::EnableIf<IsPrimitiveRuntimeReflected<T>::value>>::load(ILoadVisitor& visitor,
                                                                                  void* inst,
                                                                                  const std::string& name,
                                                                                  size_t) const
    {
        T* ptr = static_cast<T*>(inst);
        visitor(ptr, name);
    }

    template <class T>
    void TTraits<T, 9, ct::EnableIf<IsPrimitiveRuntimeReflected<T>::value>>::save(ISaveVisitor& visitor,
                                                                                  const void* inst,
                                                                                  const std::string& name,
                                                                                  size_t) const
    {
        const T* ptr = static_cast<const T*>(inst);
        visitor(ptr, name);
    }

    template <class T>
    void TTraits<T, 9, ct::EnableIf<IsPrimitiveRuntimeReflected<T>::value>>::visit(StaticVisitor& visitor,
                                                                                   const std::string&) const
    {
        visitor.template visit<T>("value");
    }

    template <class T>
    uint32_t TTraits<T, 9, ct::EnableIf<IsPrimitiveRuntimeReflected<T>::value>>::getNumMembers() const
    {
        return 0;
    }

} // namespace mo

#endif // MO_VISITATION_STRUCT_TRAITS_HPP
