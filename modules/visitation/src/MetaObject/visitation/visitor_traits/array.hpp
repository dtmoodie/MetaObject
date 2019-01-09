#ifndef MO_VISITATION_ARRAY_HPP
#define MO_VISITATION_ARRAY_HPP

#include "../TraitInterface.hpp"
#include <cassert>
namespace mo
{
    template <class T>
    struct ArrayContainerTrait : public IContainerTraits
    {
        using base = IContainerTraits;
        ArrayContainerTrait(T* ptr, const T* const_ptr, const size_t size)
            : m_ptr(ptr)
            , m_const_ptr(const_ptr)
            , m_size(size)
        {
            assert((m_ptr != nullptr) || (m_const_ptr != nullptr));
        }

        void visit(ILoadVisitor* visitor) override
        {
            if (m_ptr)
            {
                for (size_t i = 0; i < m_size; ++i)
                {
                    (*visitor)(m_ptr + i);
                }
            }
        }

        void visit(ISaveVisitor* visitor) const override
        {
            if (m_const_ptr)
            {
                for (size_t i = 0; i < m_size; ++i)
                {
                    (*visitor)(m_const_ptr + i);
                }
            }
            else
            {
                for (size_t i = 0; i < m_size; ++i)
                {
                    (*visitor)(m_ptr + i);
                }
            }
        }

        void visit(StaticVisitor* visitor) const override
        {
            visitor->visit<T>("data");
        }

        TypeInfo keyType() const override
        {
            return TypeInfo(typeid(void));
        }
        TypeInfo valueType() const override
        {
            return TypeInfo(typeid(T));
        }

        TypeInfo type() const override
        {
            return TypeInfo(typeid(T[]));
        }

        bool isContinuous() const override
        {
            return true;
        }
        bool podValues() const override
        {
            return std::is_pod<T>::value;
        }
        bool podKeys() const override
        {
            return false;
        }
        size_t getSize() const override
        {
            return m_size;
        }
        void setSize(const size_t) override
        {
        }

      private:
        T* m_ptr;
        const T* m_const_ptr;
        size_t m_size;
    };
}

#endif // MO_VISITATION_ARRAY_HPP
