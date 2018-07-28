#pragma once
#include "ct/reflect.hpp"
#include <cereal/cereal.hpp>
#include <opencv2/core/types.hpp>

namespace ct
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
    struct Reflect<cv::Matx<T, R, C>>
    {
        static constexpr int N = 0;
        static constexpr int SPECIALIZED = true;
        static constexpr MatrixAdapter<T, R, C> get(cv::Matx<T, R, C>& data, Indexer<0>) { return data.val; }
        static constexpr MatrixAdapter<const T, R, C> get(const cv::Matx<T, R, C>& data, Indexer<0>)
        {
            return data.val;
        }
        static constexpr const char* getName(Indexer<0>) { return "data"; }
        static constexpr ct::Indexer<N> end() { return ct::Indexer<N>{}; }
    };

    template <class T, int R>
    struct Reflect<cv::Vec<T, R>>
    {
        static constexpr int N = 0;
        static constexpr int SPECIALIZED = true;
        static constexpr ct::Accessor<ArrayAdapter<const T, R>(*)(const cv::Vec<T, R>&), ArrayAdapter<T,R>(*)(cv::Vec<T, R>&)> getAccessor(const ct::Indexer<0>)
        {
            return {[](const cv::Vec<T, R>& data)->ArrayAdapter<const T, R>{return &data.val[0];}, [](cv::Vec<T, R>& data)->ArrayAdapter<T, R>{return &data.val[0];}};
        }

        static constexpr ArrayAdapter<T, R> get(cv::Vec<T, R>& data, Indexer<0>) { return data.val; }
        static constexpr ArrayAdapter<const T, R> get(const cv::Vec<T, R>& data, Indexer<0>) { return data.val; }
        static constexpr const char* getName(Indexer<0>) { return "data"; }
        static constexpr ct::Indexer<N> end() { return ct::Indexer<N>{}; }
    };

    template <class T>
    struct Reflect<cv::Scalar_<T>>
    {
        static constexpr int N = 0;
        static constexpr int SPECIALIZED = true;

        static constexpr ct::Accessor<ArrayAdapter<const T, 4>(*)(const cv::Scalar_<T>&), ArrayAdapter<T, 4>(*)(cv::Scalar_<T>&)> getAccessor(const ct::Indexer<0>)
        {
            return {[](const cv::Scalar_<T>& data)->ArrayAdapter<const T, 4>{return &data.val[0];}, [](cv::Scalar_<T>& data)->ArrayAdapter<T, 4>{return &data.val[0];}};
        }

        static constexpr const char* getName(Indexer<0>) { return "data"; }
        static constexpr ct::Indexer<N> end() { return ct::Indexer<N>{}; }
    };

    REFLECT_TEMPLATED_START(cv::Rect_)
        PUBLIC_ACCESS(x)
        PUBLIC_ACCESS(y)
        PUBLIC_ACCESS(width)
        PUBLIC_ACCESS(height)
    REFLECT_END;

    REFLECT_TEMPLATED_START(cv::Point_)
        PUBLIC_ACCESS(x)
        PUBLIC_ACCESS(y)
    REFLECT_END;

    REFLECT_TEMPLATED_START(cv::Point3_)
        PUBLIC_ACCESS(x)
        PUBLIC_ACCESS(y)
        PUBLIC_ACCESS(z)
    REFLECT_END;

    REFLECT_TEMPLATED_START(cv::Size_)
        PUBLIC_ACCESS(width)
        PUBLIC_ACCESS(height)
    REFLECT_END;

}
