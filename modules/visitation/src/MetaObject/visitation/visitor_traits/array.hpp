#pragma once
#include "../IDynamicVisitor.hpp"
#include <cassert>
namespace mo
{
    template <class T>
    struct ArrayContainerTrait : public IContainerTraits
    {
        using base = IContainerTraits;
        ArrayContainerTrait(T* ptr, const T* const_ptr, const size_t size)
            : m_ptr(ptr), m_const_ptr(const_ptr), m_size(size)
        {
            assert((m_ptr != nullptr) || (m_const_ptr != nullptr));
        }

        virtual void visit(IReadVisitor* visitor) override
        {
            if (m_ptr)
            {
                for (size_t i = 0; i < m_size; ++i)
                {
                    (*visitor)(m_ptr + i);
                }
            }
        }

        virtual void visit(IWriteVisitor* visitor) const override
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

        virtual TypeInfo keyType() const override { return TypeInfo(typeid(void)); }
        virtual TypeInfo valueType() const override { return TypeInfo(typeid(T)); }

        virtual TypeInfo type() const override { return TypeInfo(typeid(T[])); }

        virtual bool isContinuous() const override { return true; }
        virtual bool podValues() const override { return std::is_pod<T>::value; }
        virtual bool podKeys() const override { return false; }
        virtual size_t getSize() const override { return m_size; }
        virtual void setSize(const size_t) override {}
        virtual std::string getName() const { return TypeInfo(typeid(T*)).name(); }

      private:
        T* m_ptr;
        const T* m_const_ptr;
        size_t m_size;
    };
}
