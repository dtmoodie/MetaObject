#pragma once
#include <cstdint>
#include <ostream>

namespace mo
{
    template<class T, int N>
    struct SmallVecStorage;
    template<class T>
    struct SmallVecDataWrapper;

    template<class T>
    struct SmallVecBase
    {
    public:
        T* begin() { return m_ptr; }
        T* end() { return m_ptr + m_size; }
        const T* begin() const { return m_ptr; }
        const T* end() const { return m_ptr + m_size; }
        size_t size() const { return m_size; }
        T& operator[](size_t i) { return begin()[i]; }
        const T& operator[](size_t i) const { return begin()[i]; }
    private:
        friend SmallVecStorage;
        friend SmallVecDataWrapper;
        T* m_ptr;
        size_t m_size;
    };

    template<class T>
    std::ostream& operator<<(std::ostream& os, const SmallVecBase<T>& vec)
    {
        const size_t size = vec.size();
        if (size)
        {
            os << '[';
            for (size_t i = 0; i < size; ++i)
            {
                if (i != 0)
                    os << ", ";
                os << vec[i];
            }
            os << ']';
        }
        return os;
    }
}