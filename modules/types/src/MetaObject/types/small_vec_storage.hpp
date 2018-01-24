#include "small_vec_base.hpp"
#include <cstring>
namespace mo
{
    template<class T, int N>
    struct SmallVecStorage
    {
        SmallVecStorage(SmallVecBase& base) :
            m_base(base)
        {

        }

        ~SmallVecStorage()
        {

        }

        void assign(T* begin, T* end)
        {
            size_t size = end - begin;
            if (m_base.m_ptr != m_data && size != m_base.m_size)
            {
                delete[] m_base.m_ptr;
            }
            if (size > N)
            {
                m_base.m_ptr = new T[size];
            }
            else 
            {
                m_base.m_ptr = m_data;
            }
            memcpy(m_base.m_ptr, begin, size * sizeof(T));
        }

    private:
        T m_data[N];
        SmallVecBase& m_base;
    };

    template<class T>
    struct SmallVecDataWrapper: public SmallVecBase<T>
    {
        SmallVecDataWrapper(T* begin, T* end)
        {
            m_ptr = begin;
            m_size = end - begin;
        }

        SmallVecDataWrapper(T* begin, size_t size)
        {
            m_ptr = begin;
            m_size = size;
        }

    };
}