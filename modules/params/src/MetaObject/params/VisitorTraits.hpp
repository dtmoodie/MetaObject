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
    auto visitValue(IReadVisitor& visitor, T& obj) -> ct::enable_if_member_setter<T, I>
    {
        auto accessor = ct::Reflect<T>::getAccessor(ct::Indexer<I>{});
        using RefType = typename ct::ReferenceType<typename decltype(accessor)::SetType>::Type;
        auto ref = static_cast<RefType>(accessor.set(obj));
        visitor(&ref, ct::Reflect<T>::getName(ct::Indexer<I>{}));
    }

    template <class T, index_t I>
    auto visitValue(IReadVisitor&, T&) -> ct::disable_if_member_setter<T, I>
    {
    }

    template <class T>
    void visitHelper(IReadVisitor& visitor, T& obj, const Indexer<0>)
    {
        visitValue<T, 0>(visitor, obj);
    }

    template <class T, index_t I>
    void visitHelper(IReadVisitor& visitor, T& obj, const Indexer<I>)
    {
        visitHelper(visitor, obj, Indexer<I - 1>{});
        visitValue<T, I>(visitor, obj);
    }

    template <class T, index_t I>
    auto visitValue(IWriteVisitor& visitor, const T& obj) -> ct::enable_if_member_getter<T, I>
    {
        auto accessor = ct::Reflect<T>::getAccessor(ct::Indexer<I>{});
        using RefType = typename ct::ReferenceType<typename decltype(accessor)::GetType>::ConstType;
        RefType ref = static_cast<RefType>(accessor.get(obj));
        visitor(&ref, ct::Reflect<T>::getName(ct::Indexer<I>{}));
    }

    template <class T, index_t I>
    auto visitValue(IWriteVisitor&, const T&) -> ct::disable_if_member_getter<T, I>
    {
    }

    template <class T>
    void visitHelper(IWriteVisitor& visitor, const T& obj, const Indexer<0>)
    {
        visitValue<T, 0>(visitor, obj);
    }

    template <class T, index_t I>
    void visitHelper(IWriteVisitor& visitor, const T& obj, const Indexer<I>)
    {
        visitHelper(visitor, obj, Indexer<I - 1>{});
        visitValue<T, I>(visitor, obj);
    }

    template <class T>
    struct TTraits<T, ct::enable_if_reflected<T>> : public IStructTraits
    {
        using base = IStructTraits;

        TTraits(T* ptr, const T* const_ptr)
            : m_ptr(ptr)
            , m_const_ptr(const_ptr)
        {
        }

        virtual void visit(IReadVisitor* visitor) override
        {
            visitHelper(*visitor, *m_ptr, ct::Reflect<T>::end());
        }

        virtual void visit(IWriteVisitor* visitor) const override
        {
            if (m_const_ptr)
            {
                visitHelper(*visitor, *m_const_ptr, ct::Reflect<T>::end());
            }
            else
            {
                visitHelper(*visitor, *m_ptr, ct::Reflect<T>::end());
            }
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
            return m_ptr ? m_ptr : m_const_ptr;
        }

        virtual void* ptr() override
        {
            return m_ptr;
        }

        virtual const char* getName() const
        {
            return ct::Reflect<T>::getName();
        }

      private:
        T* m_ptr;
        const T* m_const_ptr;
    };
}
