#pragma once
#include "IDynamicVisitor.hpp"
#include "MetaObject/detail/TypeInfo.hpp"

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
        using RefType = typename ct::ReferenceType<typename decltype(accessor)::SetType>::Type;
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
        using RefType = typename ct::ReferenceType<typename decltype(accessor)::GetType>::ConstType;
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

    template<class T, index_t I>
    auto visitValue(StaticVisitor& visitor, const Indexer<I> idx) -> ct::enable_if_member_getter<T, I>
    {
        using Type = typename decltype(ct::Reflect<T>::getAccessor(idx))::GetType;
        visitor.visit<Type>(ct::Reflect<T>::getName(idx));
    }

    template<class T, index_t I>
    auto visitValue(StaticVisitor&, const Indexer<I>) -> ct::disable_if_member_getter<T, I>
    {

    }

    template<class T>
    void visitHelper(StaticVisitor& visitor, const Indexer<0> idx)
    {
        visitValue<T>(visitor, idx);
    }

    template<class T, index_t I>
    void visitHelper(StaticVisitor& visitor, const Indexer<I> idx)
    {
        visitHelper<T>(visitor, --idx);
        visitValue<T>(visitor, idx);
    }

    template <class T>
    struct TTraits<T, ct::enable_if_reflected<T>> : public ILoadStructTraits
    {
        using base = ILoadStructTraits;

        TTraits(T* ptr, const size_t count)
            : m_ptr(ptr)
            , m_count(count)
        {
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

        void* ptr() override
        {
            return m_ptr;
        }

        std::string getName() const override
        {
            return ct::Reflect<T>::getName();
        }

        size_t count() const override
        {
            return m_count;
        }

        void increment() override{++m_ptr;}

      private:
        T* m_ptr;
        size_t m_count;
    };

    template <class T>
    struct TTraits<const T, ct::enable_if_reflected<T>> : public ISaveStructTraits
    {
        using base = ISaveStructTraits;

        TTraits(const T* ptr, const size_t count)
            : m_const_ptr(ptr)
            , m_count(count)
        {
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

        std::string getName() const override
        {
            return ct::Reflect<T>::getName();
        }

        size_t count() const override
        {
            return m_count;
        }

        void increment() override{++m_const_ptr;}

      private:
        const T* m_const_ptr;
        size_t m_count;
    };
}
