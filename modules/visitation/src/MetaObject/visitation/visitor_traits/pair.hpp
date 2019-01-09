#ifndef MO_VISITATION_PAIR_HPP
#define MO_VISITATION_PAIR_HPP
#include <MetaObject/visitation/VisitorTraits.hpp>
#include <type_traits>

namespace mo
{
    template <class T1, class T2>
    struct TTraits<std::pair<T1, T2>, void> : public ILoadStructTraits
    {
        using base = ILoadStructTraits;

        TTraits(std::pair<T1, T2>* ptr, const size_t count)
            : m_ptr(ptr)
            , m_count(count)
        {
        }

        void load(ILoadVisitor* visitor) override
        {
            (*visitor)(&m_ptr->first, "first");
            (*visitor)(&m_ptr->second, "second");
        }

        void save(ISaveVisitor* visitor) const override
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

        size_t count() const override
        {
            return m_count;
        }

        void increment() override
        {
            ++m_ptr;
        }

        void setInstance(void* ptr, const TypeInfo type_) override
        {
            MO_ASSERT(type_ == type());
            m_ptr = static_cast<std::pair<T1, T2>*>(ptr);
        }

        void setInstance(const void*, const TypeInfo) override
        {
            THROW(warn, "Attempting to set const ptr");
        }

      private:
        std::pair<T1, T2>* m_ptr;
        size_t m_count;
    };

    template <class T1, class T2>
    struct TTraits<const std::pair<T1, T2>, void> : public ISaveStructTraits
    {
        using base = ISaveStructTraits;

        TTraits(const std::pair<T1, T2>* ptr, const size_t count)
            : m_ptr(ptr)
            , m_count(count)
        {
        }

        void save(ISaveVisitor* visitor) const override
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

        TypeInfo type() const override
        {
            return TypeInfo(typeid(std::pair<T1, T2>));
        }

        size_t count() const override
        {
            return m_count;
        }

        void increment() override
        {
            ++m_ptr;
        }

        void setInstance(const void* ptr, const TypeInfo type_) override
        {
            MO_ASSERT(type_ == type());
            m_ptr = static_cast<const std::pair<T1, T2>*>(ptr);
        }

      private:
        const std::pair<T1, T2>* m_ptr;
        size_t m_count;
    };
}

#endif // MO_VISITATION_PAIR_HPP
