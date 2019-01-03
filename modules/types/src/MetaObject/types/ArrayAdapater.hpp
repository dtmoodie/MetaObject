#ifndef MO_TYPES_ARRAY_ADAPTER_HPP
#define MO_TYPES_ARRAY_ADAPTER_HPP

#include <ct/reflect.hpp>
#include <ct/TypeTraits.hpp>

#include <cereal/cereal.hpp>
#include <ostream>

namespace mo
{
    struct ISaveVisitor;
    template <class T, size_t N>
    struct ArrayAdapter
    {
        typedef T value_type;
        typedef size_t size_type;
        typedef void allocator_type;
        typedef T* iterator;
        typedef const T* const_iterator;

        ArrayAdapter(T* ptr_ = nullptr)
            : ptr(ptr_)
        {
        }

        constexpr size_t size() const
        {
            return N;
        }

        T* begin()
        {
            return ptr;
        }

        T* end()
        {
            return ptr + N;
        }

        const T* cbegin() const
        {
            return ptr;
        }

        const T* cend() const
        {
            return ptr + N;
        }

        T operator[](const size_t i) const
        {
            return ptr[i];
        }

        T& operator[](const size_t i)
        {
            return ptr[i];
        }

        T* ptr;

        template <class AR>
        typename std::enable_if<cereal::traits::is_output_serializable<cereal::BinaryData<T>, AR>::value
                          && std::is_arithmetic<T>::value>::type save(AR& ar) const
        {
            if (ptr)
            {
                for (size_t i = 0; i < N; ++i)
                {
                    ar(cereal::BinaryData<T>(ptr, N*sizeof(T)));
                }
            }
        }

        template <class AR>
        typename std::enable_if<!cereal::traits::is_output_serializable<cereal::BinaryData<T>, AR>::value
                          || !std::is_arithmetic<T>::value>::type save(AR& ar) const
        {
            if (ptr)
            {
                ar(cereal::make_size_tag(N));
                for (size_t i = 0; i < N; ++i)
                {
                    ar(ptr[i]);
                }
            }
        }

        template <class AR>
        typename std::enable_if<cereal::traits::is_input_serializable<cereal::BinaryData<T>, AR>::value
                          && std::is_arithmetic<T>::value>::type load(AR& ar)
        {
            if (ptr)
            {
                ar(cereal::BinaryData<T>(ptr, N*sizeof(T)));
            }
        }

        template <class AR>
        typename std::enable_if<!cereal::traits::is_input_serializable<cereal::BinaryData<T>, AR>::value
                          || !std::is_arithmetic<T>::value>::type load(AR& ar)
        {
            if (ptr)
            {
                ar(cereal::make_size_tag(N));
                for (int i = 0; i < N; ++i)
                {
                    ar(ptr[i]);
                }
            }
        }

    };

    template <class T, size_t N>
    std::ostream& operator<<(std::ostream& os, const ArrayAdapter<T, N>& array)
    {
        if (array.ptr)
        {
            os << "[";
            for (size_t i = 0; i < N; ++i)
            {
                if (i != 0)
                    os << ',';
                os << array.ptr[i];
            }
            os << "]";
        }
        return os;
    }

    template <size_t N>
    std::ostream& operator<<(std::ostream& os, const ArrayAdapter<const unsigned char, N>& array)
    {
        if (array.ptr)
        {
            os << "[";
            for (size_t i = 0; i < N; ++i)
            {
                if (i != 0)
                    os << ',';
                os << int(array.ptr[i]);
            }
            os << "]";
        }
        return os;
    }

    template <class T, size_t Rows, size_t Cols>
    struct MatrixAdapter
    {
        MatrixAdapter(T* ptr_ = nullptr)
            : ptr(ptr_)
        {
        }

        T& operator()(const size_t y, const size_t x)
        {
            return ptr[y * Cols + x];
        }

        const T& operator()(const size_t y, const size_t x) const
        {
            return ptr[y * Cols + x];
        }

        T* ptr;

        template <class AR>
        typename std::enable_if<cereal::traits::is_output_serializable<cereal::BinaryData<T>, AR>::value
                          && std::is_arithmetic<T>::value>::type save(AR& ar) const
        {
            if (ptr)
            {
                const size_t N = Rows * Cols;
                ar(cereal::BinaryData<T>(ptr, N*sizeof(T)));
            }
        }

        template <class AR>
        typename std::enable_if<!cereal::traits::is_output_serializable<cereal::BinaryData<T>, AR>::value
                          || !std::is_arithmetic<T>::value>::type save(AR& ar) const
        {
            if (ptr)
            {
                size_t N = Rows * Cols;
                ar(cereal::make_size_tag(N));
                for (int i = 0; i < N; ++i)
                {
                    ar(ptr[i]);
                }
            }
        }

        template <class AR>
        typename std::enable_if<cereal::traits::is_input_serializable<cereal::BinaryData<T>, AR>::value
                          && std::is_arithmetic<T>::value>::type load(AR& ar)
        {
            if (ptr)
            {
                const size_t N = Rows * Cols;
                ar(cereal::BinaryData<T>(ptr, N*sizeof(T)));
            }
        }

        template <class AR>
        typename std::enable_if<!cereal::traits::is_input_serializable<cereal::BinaryData<T>, AR>::value
                          || !std::is_arithmetic<T>::value>::type load(AR& ar)
        {
            if (ptr)
            {
                size_t N = Rows * Cols;
                ar(cereal::make_size_tag(N));
                for (int i = 0; i < N; ++i)
                {
                    ar(ptr[i]);
                }
            }
        }
    };



    template <class T, size_t ROWS, size_t COLS>
    std::ostream& operator<<(std::ostream& os, const MatrixAdapter<T, ROWS, COLS>& m)
    {
        if (m.ptr)
        {
            os << "[[";
            size_t idx = 0;
            for (size_t i = 0; i < ROWS; ++i, ++idx)
            {
                if (i != 0)
                {
                    os << "\n [";
                }
                for(size_t j = 0; j < COLS; ++j, ++idx)
                {
                    if(j != 0)
                    {
                        os << ' ';
                    }
                    os << m.ptr[idx];
                }
                os << "]";
            }
            os << "]";
        }
        return os;
    }

}

namespace ct
{
    template <class T, size_t N>
    struct ReferenceType<mo::ArrayAdapter<T, N>>
    {
        using Type = mo::ArrayAdapter<T, N>;
        using ConstType = mo::ArrayAdapter<const T, N>;
    };

    template<class T, size_t ROWS, size_t COLS>
    struct ReferenceType<mo::MatrixAdapter<T, ROWS, COLS>>
    {
        using Type = mo::MatrixAdapter<T, ROWS, COLS>;
        using ConstType = mo::MatrixAdapter<const T, ROWS, COLS>;
    };
}
#endif // MO_TYPES_ARRAY_ADAPTER_HPP
