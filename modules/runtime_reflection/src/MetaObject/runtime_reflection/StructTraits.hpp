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
        RefType ref = static_cast<RefType>(accessor.get(obj));
        auto name = ct::getName<I, T>();
        visitor(&ref, name);
        return true;
    }

    template <class T, index_t I>
    auto visitValue(ISaveVisitor&, const T&, const ct::Indexer<I> idx)
        -> ct::EnableIf<!ct::IsWritable<T, I>::value, bool>
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
    saveMemberHelper(ISaveVisitor& visitor, const T& inst, ct::Indexer<I> itr, std::string* name = nullptr)
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
        ISaveVisitor& visitor, const T& inst, const std::string& name, ct::Indexer<0> itr, uint32_t* idx = nullptr)
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

        bool loadMember(ILoadVisitor& visitor, void* inst, uint32_t idx, std::string* name = nullptr) const override
        {
            T& ref = this->ref(inst);
            const auto itr = ct::Reflect<T>::end();
            uint32_t member_counter = 0;
            return loadMemberRecurse(visitor, ref, idx, itr, member_counter, name);
        }

        bool
        saveMember(ISaveVisitor& visitor, const void* inst, uint32_t idx, std::string* name = nullptr) const override
        {
            const T& ref = this->ref(inst);
            const auto itr = ct::Reflect<T>::end();
            uint32_t member_counter = 0;
            return saveMemberRecurse(visitor, ref, idx, itr, member_counter, name);
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
    };
} // namespace mo

#endif // MO_VISITATION_STRUCT_TRAITS_HPP
