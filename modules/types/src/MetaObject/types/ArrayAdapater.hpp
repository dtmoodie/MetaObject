#ifndef MO_TYPES_ARRAY_ADAPTER_HPP
#define MO_TYPES_ARRAY_ADAPTER_HPP

#include <ct/reflect.hpp>
#include <ct/type_traits.hpp>

#include <ostream>

namespace mo
{
    struct ISaveVisitor;

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
                for (size_t j = 0; j < COLS; ++j, ++idx)
                {
                    if (j != 0)
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
} // namespace mo

namespace ct
{
    template <class T, size_t ROWS, size_t COLS>
    struct ReferenceType<mo::MatrixAdapter<T, ROWS, COLS>>
    {
        using Type = mo::MatrixAdapter<T, ROWS, COLS>;
        using ConstType = mo::MatrixAdapter<const T, ROWS, COLS>;
    };
} // namespace ct
#endif // MO_TYPES_ARRAY_ADAPTER_HPP
