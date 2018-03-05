#pragma once
#include "ct/reflect/reflect_data.hpp"
#include <cereal/cereal.hpp>
#include <opencv2/core/types.hpp>

namespace ct
{
    namespace reflect
    {
        template <class T, size_t N>
        struct ArrayAdapter
        {
            typedef T value_type;
            typedef size_t size_type;
            typedef void allocator_type;
            typedef T* iterator;
            typedef const T* const_iterator;

            ArrayAdapter(T* ptr_ = nullptr) : ptr(ptr_) {}

            constexpr size_t size() const { return N; }

            T* begin() { return ptr; }

            T* end() { return ptr + N; }

            const T* cbegin() const { return ptr; }

            const T* cend() const { return ptr + N; }

            T* ptr;

            template <class AR>
            void serialize(AR& ar)
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
            MatrixAdapter(T* ptr_ = nullptr) : ptr(ptr_) {}

            T* ptr;

            template <class AR>
            void serialize(AR& ar)
            {
                if (ptr)
                {
                    const size_t N = Rows * Cols;
                    ar(cereal::make_size_tag(N));
                    for (int i = 0; i < N; ++i)
                    {
                        ar(ptr[i]);
                    }
                }
            }
        };

        template <class T, size_t R, size_t C>
        struct ReflectData<cv::Matx<T, R, C>>
        {
            static constexpr int N = 1;
            static constexpr int IS_SPECIALIZED = true;
            static constexpr MatrixAdapter<T, R, C> get(cv::Matx<T, R, C>& data, _counter_<0>) { return data.val; }
            static constexpr MatrixAdapter<const T, R, C> get(const cv::Matx<T, R, C>& data, _counter_<0>)
            {
                return data.val;
            }
            static constexpr const char* getName(_counter_<0>) { return "data"; }
        };

        template <class T, int R>
        struct ReflectData<cv::Vec<T, R>>
        {
            static constexpr int N = 1;
            static constexpr int IS_SPECIALIZED = true;
            static constexpr ArrayAdapter<T, R> get(cv::Vec<T, R>& data, _counter_<0>) { return data.val; }
            static constexpr ArrayAdapter<const T, R> get(const cv::Vec<T, R>& data, _counter_<0>) { return data.val; }
            static constexpr const char* getName(_counter_<0>) { return "data"; }
        };

        template <class T>
        struct ReflectData<cv::Scalar_<T>>
        {
            static constexpr int N = 1;
            static constexpr int IS_SPECIALIZED = true;
            static constexpr ArrayAdapter<T, 4> get(cv::Scalar_<T>& data, _counter_<0>) { return &data.val[0]; }
            static constexpr ArrayAdapter<const T, 4> get(const cv::Scalar_<T>& data, _counter_<0>)
            {
                return &data.val[0];
            }
            static constexpr const char* getName(_counter_<0>) { return "data"; }
        };

        REFLECT_TEMPLATED_DATA_START(cv::Rect_)
            REFLECT_DATA_MEMBER(x)
            REFLECT_DATA_MEMBER(y)
            REFLECT_DATA_MEMBER(width)
            REFLECT_DATA_MEMBER(height)
        REFLECT_DATA_END;

        REFLECT_TEMPLATED_DATA_START(cv::Point_)
            REFLECT_DATA_MEMBER(x)
            REFLECT_DATA_MEMBER(y)
        REFLECT_DATA_END;

        REFLECT_TEMPLATED_DATA_START(cv::Point3_)
            REFLECT_DATA_MEMBER(x)
            REFLECT_DATA_MEMBER(y)
            REFLECT_DATA_MEMBER(z)
        REFLECT_DATA_END;

        REFLECT_TEMPLATED_DATA_START(cv::Size_)
            REFLECT_DATA_MEMBER(width)
            REFLECT_DATA_MEMBER(height)
        REFLECT_DATA_END;
    }
}
