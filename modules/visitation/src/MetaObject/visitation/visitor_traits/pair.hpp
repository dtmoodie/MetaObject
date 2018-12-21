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

        TTraits(std::pair<T1, T2>* ptr, const std::pair<T1, T2>* const_ptr)
            : m_ptr(ptr)
            , m_const_ptr(const_ptr)
        {
        }

        virtual void visit(IReadVisitor* visitor) override
        {
            (*visitor)(&m_ptr->first, "first");
            (*visitor)(&m_ptr->second, "second");
        }

        virtual void visit(IWriteVisitor* visitor) const override
        {
            if (m_const_ptr)
            {
                (*visitor)(&m_const_ptr->first, "first");
                (*visitor)(&m_const_ptr->second, "second");
            }
            else
            {
                (*visitor)(&m_ptr->first, "first");
                (*visitor)(&m_ptr->second, "second");
            }
        }

        virtual size_t size() const
        {
            return sizeof(std::pair<T1, T2>);
        }

        virtual bool triviallySerializable() const
        {
            return std::is_pod<T1>::value && std::is_pod<T2>::value;
        }

        virtual bool isPrimitiveType() const
        {
            return false;
        }

        virtual const void* ptr() const
        {
            return m_ptr ? m_ptr : m_const_ptr;
        }

        virtual void* ptr()
        {
            return m_ptr;
        }

        TypeInfo type() const override
        {
            return TypeInfo(typeid(std::pair<T1, T2>));
        }

        std::string getName() const override
        {
            return TypeInfo(typeid(std::pair<T1, T2>)).name();
        }

      private:
        std::pair<T1, T2>* m_ptr;
        const std::pair<T1, T2>* m_const_ptr;
    };
}

#endif // MO_VISITATION_PAIR_HPP
