#pragma once

#include <MetaObject/logging/logging.hpp>
#include <cstdint>
#include <ostream>
#include <cereal/cereal.hpp>

namespace mo
{
    template<class T, int N>
    struct SmallVecStorage;
    template<class T>
    struct SmallVecDataWrapper;

    template<class T>
    struct SmallVecBase
    {
        inline T* begin() { return m_ptr; }
        inline T* end() { return m_ptr + m_size; }
        inline const T* begin() const { return m_ptr; }
        inline const T* end() const { return m_ptr + m_size; }
        inline size_t size() const { return m_size; }
        inline T& operator[](size_t i) { return begin()[i]; }
        inline const T& operator[](size_t i) const { return m_ptr[i]; }
        template<class AR>
        void load(AR& ar)
        {
            size_t size;
            ar(cereal::make_size_tag(size));
            MO_ASSERT(size == m_size);
            ar(cereal::binary_data(m_ptr, m_size * sizeof(T)));
        }
        template<class AR>
        void save(AR& ar) const
        {
            if (m_size)
            {
                ar(cereal::make_size_tag(m_size));
                ar(cereal::binary_data(m_ptr, m_size * sizeof(T)));
            }
        }
    private:
        template<class T, int N>
        friend struct SmallVecStorage;
        template<class T>
        friend struct SmallVecDataWrapper;
        T* m_ptr = nullptr;
        size_t m_size = 0;
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