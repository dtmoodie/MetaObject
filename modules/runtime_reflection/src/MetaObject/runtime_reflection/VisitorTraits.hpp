#ifndef MO_VISITATION_VISITORTRAITS_HPP
#define MO_VISITATION_VISITORTRAITS_HPP
#include "TraitRegistry.hpp"

#include <MetaObject/logging/logging.hpp>

#include <ct/TypeTraits.hpp>
#include <ct/reflect.hpp>

namespace mo
{
    using index_t = ct::index_t;
    template <index_t N>
    using Indexer = ct::Indexer<N>;

    template <class T, index_t I>
    auto visitValue(ILoadVisitor& visitor, T& obj, const Indexer<I> idx) -> ct::enable_if_member_setter<T, I>
    {
        auto accessor = ct::Reflect<T>::getAccessor(idx);
        using RefType = typename ct::ReferenceType<typename decltype(accessor)::Set_t>::Type;
        RefType ref = static_cast<RefType>(accessor.set(obj));
        visitor(&ref, ct::Reflect<T>::getName(idx));
    }

    template <class T, index_t I>
    auto visitValue(ILoadVisitor&, T&, const Indexer<I>) -> ct::disable_if_member_setter<T, I>
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
        visitHelper(visitor, obj, --idx);
        visitValue(visitor, obj, idx);
    }

    template <class T, index_t I>
    auto visitValue(ISaveVisitor& visitor, const T& obj, const ct::Indexer<I> idx) -> ct::enable_if_member_getter<T, I>
    {
        auto accessor = ct::Reflect<T>::getAccessor(idx);
        using RefType = typename ct::ReferenceType<typename decltype(accessor)::Get_t>::ConstType;
        RefType ref = static_cast<RefType>(accessor.get(obj));
        visitor(&ref, ct::Reflect<T>::getName(idx));
    }

    template <class T, index_t I>
    auto visitValue(ISaveVisitor&, const T&, const ct::Indexer<I>) -> ct::disable_if_member_getter<T, I>
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
        visitHelper(visitor, obj, --idx);
        visitValue(visitor, obj, idx);
    }

    template <class T, index_t I>
    auto visitValue(StaticVisitor& visitor, const Indexer<I> idx) -> ct::enable_if_member_getter<T, I>
    {
        using Type = typename decltype(ct::Reflect<T>::getAccessor(idx))::Get_t;
        visitor.visit<Type>(ct::Reflect<T>::getName(idx));
    }

    template <class T, index_t I>
    auto visitValue(StaticVisitor&, const Indexer<I>) -> ct::disable_if_member_getter<T, I>
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
        visitHelper<T>(visitor, --idx);
        visitValue<T>(visitor, idx);
    }

    template <class T>
    struct TTraits<T, ct::enable_if_reflected<T>> : public ILoadStructTraits
    {
        using base = ILoadStructTraits;
        static mo::TraitRegisterer<T> reg;

        TTraits(T* ptr, const size_t count)
            : m_ptr(ptr)
            , m_count(count)
        {
            (void)reg;
        }

        void load(ILoadVisitor* visitor) override
        {
            visitHelper(*visitor, *m_ptr, ct::Reflect<T>::end());
        }

        void save(ISaveVisitor* visitor) const override
        {
            visitHelper(*visitor, *m_ptr, ct::Reflect<T>::end());
        }

        void visit(StaticVisitor* visitor) const override
        {
            visitHelper<T>(*visitor, ct::Reflect<T>::end());
        }

        bool isPrimitiveType() const override
        {
            return false;
        }

        size_t size() const override
        {
            return sizeof(T);
        }

        TypeInfo type() const override
        {
            return TypeInfo(typeid(T));
        }

        bool triviallySerializable() const override
        {
            return std::is_pod<T>::value;
        }

        const void* ptr() const override
        {
            return m_ptr;
        }

        void setInstance(const void*, const TypeInfo) override
        {
            MO_LOG(warn, "Const casting a const void*");
        }

        void* ptr() override
        {
            return m_ptr;
        }

        void setInstance(void* ptr, const TypeInfo type_) override
        {
            MO_ASSERT(type_ == type());
            m_ptr = static_cast<T*>(ptr);
        }

        std::string getName() const override
        {
            return ct::Reflect<T>::getName();
        }

        size_t count() const override
        {
            return m_count;
        }

        void increment() override
        {
            ++m_ptr;
        }

      private:
        T* m_ptr;
        size_t m_count;
    };

    template <class T>
    struct TTraits<const T, ct::enable_if_reflected<T>> : public ISaveStructTraits
    {
        using base = ISaveStructTraits;
        static mo::TraitRegisterer<const T> reg;

        TTraits(const T* ptr = nullptr, const size_t count = 0)
            : m_const_ptr(ptr)
            , m_count(count)
        {
            (void)reg;
        }

        void save(ISaveVisitor* visitor) const override
        {
            if (m_const_ptr)
            {
                visitHelper(*visitor, *m_const_ptr, ct::Reflect<T>::end());
            }
        }

        void visit(StaticVisitor* visitor) const override
        {
            visitHelper<T>(*visitor, ct::Reflect<T>::end());
        }

        bool isPrimitiveType() const override
        {
            return false;
        }

        size_t size() const override
        {
            return sizeof(T);
        }

        TypeInfo type() const override
        {
            return TypeInfo(typeid(T));
        }

        bool triviallySerializable() const override
        {
            return std::is_pod<T>::value;
        }

        const void* ptr() const override
        {
            return m_const_ptr;
        }

        void setInstance(const void* ptr, const TypeInfo type_) override
        {
            MO_ASSERT(type_ == type());
            m_const_ptr = static_cast<const T*>(ptr);
        }

        std::string getName() const override
        {
            return ct::Reflect<T>::getName();
        }

        size_t count() const override
        {
            return m_count;
        }

        void increment() override
        {
            ++m_const_ptr;
        }

      private:
        const T* m_const_ptr;
        size_t m_count;
    };

    template <class T>
    TraitRegisterer<T> TTraits<T, ct::enable_if_reflected<T>>::reg;

    template <class T>
    TraitRegisterer<const T> TTraits<const T, ct::enable_if_reflected<T>>::reg;
}

#endif // MO_VISITATION_VISITORTRAITS_HPP
