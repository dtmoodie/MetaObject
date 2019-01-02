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
    auto visitValue(IReadVisitor& visitor, T& obj, const Indexer<I> idx) -> ct::enable_if_member_setter<T, I>
    {
        auto accessor = ct::Reflect<T>::getAccessor(idx);
        using RefType = typename ct::ReferenceType<typename decltype(accessor)::SetType>::Type;
        RefType ref = static_cast<RefType>(accessor.set(obj));
        visitor(&ref, ct::Reflect<T>::getName(idx));
    }

    template <class T, index_t I>
    auto visitValue(IReadVisitor&, T&, const Indexer<I>) -> ct::disable_if_member_setter<T, I>
    {
    }

    template <class T>
    void visitHelper(IReadVisitor& visitor, T& obj, const Indexer<0> idx)
    {
        visitValue(visitor, obj, idx);
    }

    template <class T, index_t I>
    void visitHelper(IReadVisitor& visitor, T& obj, const Indexer<I> idx)
    {
        visitHelper(visitor, obj, --idx);
        visitValue(visitor, obj, idx);
    }

    template <class T, index_t I>
    auto visitValue(IWriteVisitor& visitor, const T& obj, const ct::Indexer<I> idx) -> ct::enable_if_member_getter<T, I>
    {
        auto accessor = ct::Reflect<T>::getAccessor(idx);
        using RefType = typename ct::ReferenceType<typename decltype(accessor)::GetType>::ConstType;
        RefType ref = static_cast<RefType>(accessor.get(obj));
        visitor(&ref, ct::Reflect<T>::getName(idx));
    }

    template <class T, index_t I>
    auto visitValue(IWriteVisitor&, const T&, const ct::Indexer<I>) -> ct::disable_if_member_getter<T, I>
    {
    }

    template <class T>
    void visitHelper(IWriteVisitor& visitor, const T& obj, const Indexer<0> idx)
    {
        visitValue(visitor, obj, idx);
    }

    template <class T, index_t I>
    void visitHelper(IWriteVisitor& visitor, const T& obj, const Indexer<I> idx)
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
    struct TTraits<T, ct::enable_if_reflected<T>> : public IStructTraits
    {
        using base = IStructTraits;

        TTraits(T* ptr)
            : m_ptr(ptr)
        {
        }

        virtual void visit(IReadVisitor* visitor) override
        {
            visitHelper(*visitor, *m_ptr, ct::Reflect<T>::end());
        }

        virtual void visit(IWriteVisitor* visitor) const override
        {
            visitHelper(*visitor, *m_ptr, ct::Reflect<T>::end());
        }

        virtual void visit(StaticVisitor* visitor) const override
        {
            visitHelper<T>(*visitor, ct::Reflect<T>::end());
        }

        virtual bool isPrimitiveType() const override
        {
            return false;
        }

        virtual size_t size() const override
        {
            return sizeof(T);
        }

        virtual TypeInfo type() const
        {
            return TypeInfo(typeid(T));
        }

        virtual bool triviallySerializable() const override
        {
            return std::is_pod<T>::value;
        }

        virtual const void* ptr() const override
        {
            return m_ptr;
        }

        virtual void* ptr() override
        {
            return m_ptr;
        }

        virtual std::string getName() const
        {
            return ct::Reflect<T>::getName();
        }

      private:
        T* m_ptr;
    };

    template <class T>
    struct TTraits<const T, ct::enable_if_reflected<T>> : public IStructTraits
    {
        using base = IStructTraits;

        TTraits(const T* ptr)
            : m_const_ptr(ptr)
        {
        }

        virtual void visit(IReadVisitor* ) override
        {

        }

        virtual void visit(IWriteVisitor* visitor) const override
        {
            if (m_const_ptr)
            {
                visitHelper(*visitor, *m_const_ptr, ct::Reflect<T>::end());
            }
        }

        virtual void visit(StaticVisitor* visitor) const override
        {
            visitHelper<T>(*visitor, ct::Reflect<T>::end());
        }

        virtual bool isPrimitiveType() const override
        {
            return false;
        }

        virtual size_t size() const override
        {
            return sizeof(T);
        }

        virtual TypeInfo type() const
        {
            return TypeInfo(typeid(T));
        }

        virtual bool triviallySerializable() const override
        {
            return std::is_pod<T>::value;
        }

        virtual const void* ptr() const override
        {
            return m_const_ptr;
        }

        virtual void* ptr() override
        {
            return nullptr;
        }

        virtual std::string getName() const
        {
            return ct::Reflect<T>::getName();
        }

      private:
        const T* m_const_ptr;
    };
}
