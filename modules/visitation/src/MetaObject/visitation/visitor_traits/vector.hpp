#pragma once
#include "../DynamicVisitor.hpp"

namespace mo
{
    template <class T>
    struct TTraits<std::vector<T>, void> : public IContainerTraits
    {
        using base = IContainerTraits;

        TTraits(std::vector<T>* vec, const std::vector<T>* const_vec) : m_vec(vec), m_const_vec(const_vec) {}

        virtual void visit(IReadVisitor* visitor) override
        {
            if (IsPrimitive<T>::value)
            {
                (*visitor)(m_vec->data(), "", m_vec->size());
            }
            else
            {
                for (size_t i = 0; i < m_vec->size(); ++i)
                {
                    (*visitor)(&(*m_vec)[i]);
                }
            }
        }

        virtual void visit(IWriteVisitor* visitor) const override
        {
            if (m_const_vec)
            {
                if (IsPrimitive<T>::value)
                {
                    (*visitor)(m_const_vec->data(), "", m_const_vec->size());
                }
                else
                {
                    for (size_t i = 0; i < m_const_vec->size(); ++i)
                    {
                        (*visitor)(&(*m_const_vec)[i]);
                    }
                }
            }
            else
            {
                if (IsPrimitive<T>::value)
                {
                    (*visitor)(m_vec->data(), "", m_vec->size());
                }
                else
                {
                    for (size_t i = 0; i < m_vec->size(); ++i)
                    {
                        (*visitor)(&(*m_vec)[i]);
                    }
                }
            }
        }

        virtual TypeInfo keyType() const override { return TypeInfo(typeid(void)); }
        virtual TypeInfo valueType() const override { return TypeInfo(typeid(T)); }
        virtual TypeInfo type() const { return TypeInfo(typeid(std::vector<T>)); }
        virtual bool isContinuous() const override { return true; }
        virtual bool podValues() const override { return std::is_pod<T>::value; }
        virtual bool podKeys() const override { return false; }

        virtual size_t getSize() const override { return (m_vec ? m_vec->size() : m_const_vec->size()); }
        virtual void setSize(const size_t num) override { m_vec->resize(num); }

        virtual std::string getName() const { return TypeInfo(typeid(std::vector<T>)).name(); }
      private:
        std::vector<T>* m_vec;
        const std::vector<T>* m_const_vec;
    };
}
