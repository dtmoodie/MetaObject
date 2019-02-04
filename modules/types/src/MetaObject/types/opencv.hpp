#ifndef MO_TYPES_OPENCV_HPP
#define MO_TYPES_OPENCV_HPP
#include "ArrayAdapater.hpp"

#include <opencv2/core/types.hpp>

namespace ct
{
    template <class T, int R>
    struct Reflect<cv::Vec<T, R>>
    {
        using DataType = cv::Vec<T, R>;
        static constexpr int NUM_FIELDS = 1;
        static constexpr int SPECIALIZED = true;

        using PropertyPointer = ct::MemberPropertyPointer<mo::ArrayAdapter<const T, R>(*)(const DataType&), mo::ArrayAdapter<T, R>(*)(DataType&)>;

        static constexpr auto getPtr(const ct::Indexer<0>) ->PropertyPointer
        {
            return PropertyPointer("data", [](const cv::Vec<T, R>& data) -> mo::ArrayAdapter<const T, R> { return &data.val[0]; },
                    [](cv::Vec<T, R>& data) -> mo::ArrayAdapter<T, R> { return &data.val[0]; });
        }

        static constexpr mo::ArrayAdapter<T, R> get(cv::Vec<T, R>& data, Indexer<0>)
        {
            return data.val;
        }
        static constexpr mo::ArrayAdapter<const T, R> get(const cv::Vec<T, R>& data, Indexer<0>)
        {
            return data.val;
        }
        static constexpr const char* getName(Indexer<0>)
        {
            return "data";
        }
        static constexpr const char* getName()
        {
            return "cv::Vec_<T, R>";
        }
        static constexpr ct::Indexer<0> end()
        {
            return ct::Indexer<0>{};
        }
    };

    template <class T>
    struct Reflect<cv::Scalar_<T>>
    {
        using DataType = cv::Scalar_<T>;
        static constexpr int NUM_FIELDS = 1;
        static constexpr int SPECIALIZED = true;
        using PropertyPointer = ct::MemberPropertyPointer<mo::ArrayAdapter<const T, 4>(*)(const DataType&), mo::ArrayAdapter<T, 4>(*)(DataType&)>;

        static constexpr auto getPtr(const ct::Indexer<0>) ->
            PropertyPointer
        {
            return PropertyPointer("data", [](const DataType& data) -> mo::ArrayAdapter<const T, 4> { return &data.val[0]; },
                    [](DataType& data) -> mo::ArrayAdapter<T, 4> { return &data.val[0]; });
        }

        static constexpr const char* getName()
        {
            return "cv::Scalar_<T>";
        }

        static constexpr ct::Indexer<0> end()
        {
            return ct::Indexer<0>{};
        }
    };

    template <class T, int ROWS, int COLS>
    struct Reflect<cv::Matx<T, ROWS, COLS>>
    {
        using DataType = cv::Matx<T, ROWS, COLS>;
        static constexpr int NUM_FIELDS = 1;
        static constexpr int SPECIALIZED = true;

        using Mat_t = cv::Matx<T, ROWS, COLS>;

        using ConstAdapater_t = mo::MatrixAdapter<const T, ROWS, COLS>;
        using Adapter_t = mo::MatrixAdapter<T, ROWS, COLS>;

        using PropertyPointer_t = ct::MemberPropertyPointer<ConstAdapater_t(*)(const DataType&), Adapter_t(*)(DataType&)>;

        static constexpr PropertyPointer_t getPtr(const ct::Indexer<0>)
        {
            return PropertyPointer_t("data", [](const Mat_t& data) -> ConstAdapater_t { return &data.val[0]; },
                              [](Mat_t& data) -> Adapter_t { return &data.val[0]; });
        }

        static constexpr const char* getName()
        {
            return "cv::Matx_<T, ROWS, COLS>";
        }

        static constexpr ct::Indexer<0> end()
        {
            return ct::Indexer<0>{};
        }
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
#endif // MO_TYPES_OPENCV_HPP
