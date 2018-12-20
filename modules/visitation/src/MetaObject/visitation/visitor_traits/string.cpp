#include "string.hpp"

namespace mo
{
    TTraits<std::string, void>::TTraits(std::string* ptr, const std::string* const_ptr)
        : m_ptr(ptr)
        , m_const_ptr(const_ptr)
    {
    }

    void TTraits<std::string, void>::visit(IReadVisitor* visitor)
    {
        (*visitor)(&(*m_ptr)[0], "", m_ptr->size());
    }

    void TTraits<std::string, void>::visit(IWriteVisitor* visitor) const
    {
        if (m_const_ptr)
        {
            (*visitor)(&(*m_const_ptr)[0], "", m_const_ptr->size());
        }
        else
        {
            (*visitor)(&(*m_ptr)[0], "", m_ptr->size());
        }
    }

    TypeInfo TTraits<std::string, void>::keyType() const
    {
        return TypeInfo(typeid(void));
    }

    TypeInfo TTraits<std::string, void>::valueType() const
    {
        return TypeInfo(typeid(char));
    }

    TypeInfo TTraits<std::string, void>::type() const
    {
        return TypeInfo(typeid(std::string));
    }

    bool TTraits<std::string, void>::isContinuous() const
    {
        return true;
    }

    bool TTraits<std::string, void>::podValues() const
    {
        return true;
    }

    bool TTraits<std::string, void>::podKeys() const
    {
        return false;
    }

    size_t TTraits<std::string, void>::getSize() const
    {
        return (m_ptr ? m_ptr->size() : m_const_ptr->size());
    }

    void TTraits<std::string, void>::setSize(const size_t num)
    {
        m_ptr->resize(num);
    }

    const char* TTraits<std::string, void>::getName() const
    {
        return "std::string";
    }
}
