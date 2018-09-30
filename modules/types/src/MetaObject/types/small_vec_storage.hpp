#pragma once
#include "small_vec_base.hpp"
#include <cstring>

namespace mo
{
    template <class T, int N>
    struct SmallVecStorage
    {
        SmallVecStorage(SmallVecBase<T>& base) : m_base(base) { m_base.m_ptr = m_data; }

        ~SmallVecStorage()
        {
            if (m_base.m_ptr != m_data)
            {
                delete[] m_base.m_ptr;
            }
        }

        void resize(size_t size)
        {
            if (m_base.m_ptr != m_data && size != m_base.m_size)
            {
                delete[] m_base.m_ptr;
            }
            if (size > N)
            {
                m_base.m_ptr = static_cast<T*>(std::malloc(size * sizeof(T)));
            }
            else
            {
                m_base.m_ptr = m_data;
            }
            m_base.m_size = size;
        }

        void assign(const T* begin, const T* end)
        {
            const size_t size = end - begin;
            resize(size);
            memcpy(m_base.m_ptr, begin, size * sizeof(T));
        }

        void append(const T& data)
        {
            if (m_base.m_size < N)
            {
                m_data[m_base.m_size] = data;
                ++m_base.m_size;
            }
            else
            {
                // allocate memory
            }
        }
        void append(T&& data)
        {
            if (m_base.m_size < N)
            {
                m_data[m_base.m_size] = std::move(data);
                ++m_base.m_size;
            }
            else
            {
                // allocate memory
            }
        }

      private:
        T m_data[N];
        SmallVecBase<T>& m_base;
    };

    template <class T>
    struct SmallVecDataWrapper : public SmallVecBase<T>
    {
        SmallVecDataWrapper(T* begin, T* end)
        {
            this->m_ptr = begin;
            this->m_size = end - begin;
        }

        SmallVecDataWrapper(T* begin, size_t size)
        {
            this->m_ptr = begin;
            this->m_size = size;
        }
    };
}
