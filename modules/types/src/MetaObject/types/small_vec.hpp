#pragma once
#include "MetaObject/types.hpp"
#include "small_vec_storage.hpp"

#include <ct/reflect.hpp>
#include <ct/reflect_macros.hpp>
#include <ct/types/TArrayView.hpp>

#include <vector>

namespace mo
{
    template <class T, int N>
    struct MO_EXPORTS SmallVec
    {
        SmallVec()
        {
        }

        SmallVec(const T& obj)
        {
            assign(&obj, &obj + 1);
        }

        SmallVec(const SmallVec& other)
        {
            assign(other.begin(), other.end());
        }

        template <int N1>
        SmallVec(const SmallVec<T, N1>& other)
        {
            assign(other.begin(), other.end());
        }

        template <int N1>
        SmallVec(SmallVec<T, N1>&& other)
        {
            if (other.m_data != other.m_ptr)
            {
                m_ptr = other.m_ptr;
                m_size = other.m_size;
                other.m_ptr = other.m_data;
                other.m_size = 0;
            }
            else
            {
                assign(other.begin(), other.end());
            }
            m_owns_data = true;
        }

        SmallVec(const std::vector<T>& vec)
        {
            assign(vec.begin(), vec.end());
        }

        ~SmallVec()
        {
            if (m_ptr != m_data && m_owns_data)
            {
                delete[] m_ptr;
                m_ptr = nullptr;
            }
        }

        SmallVec<T, N>& operator=(const std::vector<T>& vec)
        {
            assign(&(*vec.begin()), &(*vec.end()));
            return *this;
        }

        template <int N1>
        SmallVec<T, N>& operator=(const SmallVec<T, N1>& vec)
        {
            assign(vec.begin(), vec.end());
            return *this;
        }

        SmallVec<T, N>& operator=(const SmallVec& vec)
        {
            assign(vec.begin(), vec.end());
            return *this;
        }

        SmallVec<T, N>& operator=(const T& obj)
        {
            assign(&obj, &obj + 1);
            return *this;
        }

        template <class T1, int N1>
        SmallVec<T, N>& operator=(SmallVec<T1, N1>&& vec)
        {
            if (vec.m_data != vec.m_ptr)
            {
                m_ptr = vec.m_ptr;
                m_size = vec.m_size;
                vec.m_ptr = vec.m_data;
                vec.m_size = 0;
            }
            else
            {
                assign(vec.begin(), vec.end());
            }
            return *this;
        }

        void resize(unsigned char size)
        {
            if (size > N)
            {
                if (m_ptr != m_data)
                {
                    T* current = m_ptr;
                    m_ptr = static_cast<T*>(std::malloc(size * sizeof(T)));
                    if (m_size)
                        std::memcpy(m_ptr, current, m_size * sizeof(T));
                    m_size = size;
                    if (m_owns_data)
                    {
                        delete[] current;
                    }
                    m_owns_data = true;
                }
                else
                {
                    m_ptr = static_cast<T*>(std::malloc(size * sizeof(T)));
                    if (m_size)
                    {
                        std::memcpy(m_ptr, m_data, m_size * sizeof(T));
                    }
                    m_size = size;
                    m_owns_data = true;
                }
            }
            else
            {
                if (m_ptr != m_data)
                {
                    if (m_size)
                        std::memcpy(m_data, m_ptr, std::min<int>(m_size, N) * sizeof(T));
                    if (m_owns_data)
                        delete[] m_ptr;
                    m_ptr = m_data;
                }
                m_size = size;
            }
        }

        void assign(const T* begin, const T* end)
        {
            resize(end - begin);
            std::memcpy(m_ptr, begin, (end - begin) * sizeof(T));
        }

        void wrap(T* begin, T* end)
        {
            if (m_ptr != m_data && m_owns_data)
                delete[] m_ptr;
            m_ptr = begin;
            m_size = end - begin;
            m_owns_data = false;
        }

        void append(const T& data)
        {
            if (m_size + 1 < N)
            {
                m_data[m_size] = data;
                ++m_size;
            }
            else
            {
                resize(m_size + 1);
                m_ptr[m_size] = data;
                ++m_size;
            }
        }

        void append(T&& data)
        {
            if (m_size < N)
            {
                m_data[m_size] = std::move(data);
                ++m_size;
            }
            else
            {
                resize(m_size + 1);
                m_ptr[m_size] = std::move(data);
                ++m_size;
            }
        }

        void pop_back()
        {
            --m_size;
        }

        void erase(int i)
        {
            for (i = i + 1; i < m_size; ++i)
            {
                m_ptr[i - 1] = m_ptr[i];
            }
            m_size--;
        }

        T* begin()
        {
            return m_ptr;
        }
        T* end()
        {
            return m_ptr + m_size;
        }
        const T* begin() const
        {
            return m_ptr;
        }
        const T* end() const
        {
            return m_ptr + m_size;
        }
        unsigned char size() const
        {
            return m_size;
        }
        const T& operator[](int i) const
        {
            if (i >= 0)
            {
                return begin()[i];
            }
            else
            {
                return end()[i];
            }
        }
        T& operator[](int i)
        {
            if (i >= 0)
            {
                return begin()[i];
            }
            else
            {
                return end()[i];
            }
        }

        template <class AR>
        void load(AR& ar)
        {
            size_t size;
            ar(cereal::make_size_tag(size));
            resize(size);
            for (size_t i = 0; i < m_size; ++i)
            {
                ar(m_ptr[i]);
            }
        }

        template <class AR>
        void save(AR& ar) const
        {
            if (m_size)
            {
                ar(cereal::make_size_tag(m_size));
                for (size_t i = 0; i < m_size; ++i)
                {
                    ar(m_ptr[i]);
                }
            }
        }

      private:
        T* m_ptr = m_data;
        unsigned char m_size = 0;
        bool m_owns_data = true;
        T m_data[N];
    };

    template <class T, int N>
    std::ostream& operator<<(std::ostream& os, const SmallVec<T, N>& vec)
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
} // namespace mo

namespace ct
{
    template <class T, int N>
    struct ReflectImpl<mo::SmallVec<T, N>>
    {
        using DataType = mo::SmallVec<T, N>;
        using this_t = ReflectImpl<DataType, void>;

        static constexpr StringView getTypeName()
        {
            return GetName<DataType>::getName();
        }

        static TArrayView<const T> getData(const DataType& arr)
        {
            return {arr.begin(), static_cast<size_t>(arr.size())};
        }

        static TArrayView<T> getDataMutable(DataType& arr)
        {
            return {arr.begin(), static_cast<size_t>(arr.size())};
        }

        REFLECT_STUB
            PROPERTY(size, &DataType::size, &DataType::resize)
            PROPERTY(data, &this_t::getData, &this_t::getDataMutable)
        REFLECT_INTERNAL_END;

        static constexpr Indexer<NUM_FIELDS - 1> end()
        {
            return Indexer<NUM_FIELDS - 1>();
        }
    };
} // namespace ct