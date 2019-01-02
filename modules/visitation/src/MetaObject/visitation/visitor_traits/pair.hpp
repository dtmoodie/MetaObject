#ifndef MO_VISITATION_PAIR_HPP
#define MO_VISITATION_PAIR_HPP
#include <MetaObject/visitation/VisitorTraits.hpp>
#include <type_traits>

namespace mo
{
    template <class T1, class T2>
    struct TTraits<std::pair<T1, T2>, void> : public IStructTraits
    {
        using base = IStructTraits;

        TTraits(std::pair<T1, T2>* ptr)
            : m_ptr(ptr)
        {
        }

        void visit(IReadVisitor* visitor) override
        {
            (*visitor)(&m_ptr->first, "first");
            (*visitor)(&m_ptr->second, "second");
        }

        void visit(IWriteVisitor* visitor) const override
        {
            (*visitor)(&m_ptr->first, "first");
            (*visitor)(&m_ptr->second, "second");
        }

        void visit(StaticVisitor* visitor) const override
        {
            visitor->template visit<T1>("first");
            visitor->template visit<T2>("second");
        }

        size_t size() const override
        {
            return sizeof(std::pair<T1, T2>);
        }

        bool triviallySerializable() const override
        {
            return std::is_pod<T1>::value && std::is_pod<T2>::value;
        }

        bool isPrimitiveType() const override
        {
            return false;
        }

        const void* ptr() const override
        {
            return m_ptr;
        }

        void* ptr() override
        {
            return m_ptr;
        }

        TypeInfo type() const override
        {
            return TypeInfo(typeid(std::pair<T1, T2>));
        }

      private:
        std::pair<T1, T2>* m_ptr;
    };

    template <class T1, class T2>
    struct TTraits<const std::pair<T1, T2>, void> : public IStructTraits
    {
        using base = IStructTraits;

        TTraits(const std::pair<T1, T2>* ptr)
            : m_ptr(ptr)
        {
        }

        void visit(IReadVisitor* visitor) override
        {
            (*visitor)(&m_ptr->first, "first");
            (*visitor)(&m_ptr->second, "second");
        }

        void visit(IWriteVisitor* visitor) const override
        {
                (*visitor)(&m_ptr->first, "first");
                (*visitor)(&m_ptr->second, "second");
        }

        void visit(StaticVisitor* visitor) const override
        {
            visitor->template visit<T1>("first");
            visitor->template visit<T2>("second");
        }

        size_t size() const override
        {
            return sizeof(std::pair<T1, T2>);
        }

        bool triviallySerializable() const override
        {
            return std::is_pod<T1>::value && std::is_pod<T2>::value;
        }

        bool isPrimitiveType() const override
        {
            return false;
        }

        const void* ptr() const override
        {
            return m_ptr;
        }

        void* ptr() override
        {
            return nullptr;
        }

        TypeInfo type() const override
        {
            return TypeInfo(typeid(std::pair<T1, T2>));
        }

      private:
        const std::pair<T1, T2>* m_ptr;
    };
}

#endif // MO_VISITATION_PAIR_HPP
