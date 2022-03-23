#pragma once
#ifdef MO_HAVE_OPENCV
#include <MetaObject/types/opencv.hpp>
#include <opencv2/core.hpp>
#endif

#include <MetaObject/runtime_reflection.hpp>

#include <MetaObject/core/TypeTable.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/array_adapter.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/map.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/memory.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/vector.hpp>

#include <ct/reflect.hpp>
#include <ct/reflect/compare.hpp>
#include <ct/reflect/print-container-inl.hpp>

#include <cereal/types/map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>

#include <iostream>
#include <memory>

#include <gtest/gtest.h>

namespace mo
{
    struct PrintVisitor : public mo::StaticVisitor
    {
        int m_indent = 2;

        PrintVisitor(std::ostream& os)
            : m_os(os)
        {
        }

        void visit(const mo::ITraits* trait, const std::string& name, const size_t cnt = 1) override
        {
            // TODO account for cnt > 1
            m_indent += 2;
            m_os << name << ": " << trait->name() << std::endl;
            m_path.push_back(trait->name());
            trait->visit(*this, name);
            m_indent -= 2;
            m_path.pop_back();
        }

        void implDyn(const mo::TypeInfo type, const std::string& name, const size_t cnt) override
        {
            // TODO account for cnt > 1
            indent();
            m_os << name << ": " << mo::TypeTable::instance()->typeToName(type) << std::endl;
            m_items.push_back(getName() + "." + name);
        }

        std::string getName()
        {
            std::stringstream ss;
            for (const std::string& path : m_path)
            {
                ss << path;
                ss << '.';
            }
            return std::move(ss).str();
        }

        void indent()
        {
            for (int i = 0; i < m_indent; ++i)
            {
                m_os << ' ';
            }
        }

        std::vector<std::string> m_path;
        std::vector<std::string> m_items;

        std::ostream& m_os;
    };

    template <class T, size_t ROWS, size_t COLS>
    bool operator==(const mo::MatrixAdapter<T, ROWS, COLS>& lhs, const mo::MatrixAdapter<T, ROWS, COLS>& rhs)
    {
        for (int i = 0; i < ROWS; ++i)
        {
            for (int j = 0; j < COLS; ++j)
            {
                if (lhs(i, j) != rhs(i, j))
                {
                    return false;
                }
            }
        }
        return true;
    }

    template <class T, size_t ROWS, size_t COLS>
    bool operator==(const mo::MatrixAdapter<const T, ROWS, COLS>& lhs,
                    const mo::MatrixAdapter<const T, ROWS, COLS>& rhs)
    {
        for (int i = 0; i < ROWS; ++i)
        {
            for (int j = 0; j < COLS; ++j)
            {
                if (lhs(i, j) != rhs(i, j))
                {
                    return false;
                }
            }
        }
        return true;
    }

    template <class T, ssize_t N>
    bool operator==(const ct::TArrayView<T, N>& lhs, const ct::TArrayView<T, N>& rhs)
    {
        if (rhs.size() != lhs.size())
        {
            return false;
        }
        for (size_t i = 0; i < lhs.size(); ++i)
        {
            if (lhs[i] != rhs[i])
            {
                return false;
            }
        }
        return true;
    }

    template <class K, class V>
    bool operator==(const mo::KVP<K, V>& lhs, const mo::KVP<K, V>& rhs)
    {
        return lhs.key == rhs.key && lhs.value == rhs.value;
    }

    template <class K, class V>
    std::ostream& operator<<(std::ostream& os, const mo::KVP<K, V>& rhs)
    {
        os << rhs.key << ":" << rhs.value;
        return os;
    }
} // namespace mo

struct DebugEqual
{
    template <class T>
    bool test(const char*, const T& lhs, const T& rhs) const
    {
        EXPECT_EQ(lhs, rhs);
        return true;
    }

    template <class T>
    bool test(const char*, const std::shared_ptr<T>& lhs, const std::shared_ptr<T>& rhs) const
    {
        EXPECT_EQ(lhs, rhs);
        return true;
    }

    template <class T>
    bool test(const T& lhs, const T& rhs) const
    {
        EXPECT_EQ(lhs, rhs);
        return true;
    }

    template <class T>
    bool test(const std::shared_ptr<T>& lhs, const std::shared_ptr<T>& rhs) const
    {
        EXPECT_EQ(lhs, rhs);
        return true;
    }
};

struct TestPodStruct
{
    float x, y, z;
    uint32_t id;
};

struct Vec
{
    float x, y, z;
};

struct VecOwner
{
    std::vector<Vec> my_vecs;
};

namespace std
{
    template <class K, class V>
    ostream& operator<<(ostream& os, const map<K, V>& m)
    {
        ct::printStruct(os, m);
        return os;
    }
} // namespace std

namespace ct
{
    REFLECT_BEGIN(TestPodStruct)
        PUBLIC_ACCESS(x)
        PUBLIC_ACCESS(y)
        PUBLIC_ACCESS(z)
        PUBLIC_ACCESS(id)
    REFLECT_END;

    REFLECT_BEGIN(Vec)
        PUBLIC_ACCESS(x)
        PUBLIC_ACCESS(y)
        PUBLIC_ACCESS(z)
    REFLECT_END;

    REFLECT_BEGIN(VecOwner)
        PUBLIC_ACCESS(my_vecs)
    REFLECT_END;
} // namespace ct

struct EqualHelper
{
    template <typename T1, typename T2>
    static testing::AssertionResult
    Compare(const char* lhs_expression, const char* rhs_expression, const T1& lhs, const T2& rhs)
    {
        return testing::internal::EqHelper::Compare(lhs_expression, rhs_expression, lhs, rhs);
    }
};

template <class T>
struct TestData;

#define TEST_DATA(TYPE, ...)                                                                                           \
    template <>                                                                                                        \
    struct TestData<TYPE> : EqualHelper                                                                                \
    {                                                                                                                  \
        static TYPE init()                                                                                             \
        {                                                                                                              \
            return __VA_ARGS__;                                                                                        \
        }                                                                                                              \
    }

TEST_DATA(int, 10);
TEST_DATA(float, 13.2F);
TEST_DATA(double, 3.14159);
TEST_DATA(std::string, "asdf");
TEST_DATA(std::vector<std::string>, {"asdf", "1235"});
TEST_DATA(TestPodStruct, {0, 1, 2, 10});
TEST_DATA(std::vector<Vec>, {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}});
TEST_DATA(VecOwner, {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}});
// TEST_DATA(std::shared_ptr<Vec>, std::make_shared<Vec>(1.0, 2.0, 3.0));
template <>
struct TestData<std::shared_ptr<Vec>> : EqualHelper
{
    static std::shared_ptr<Vec> init()
    {
        auto ret = std::make_shared<Vec>();
        ret->x = 1.0;
        ret->y = 2.0;
        ret->z = 3.0;
        return ret;
    }
};
template <>
struct TestData<std::map<std::string, Vec>> : EqualHelper
{
    static std::map<std::string, Vec> init()
    {
        std::map<std::string, Vec> vec_map;
        vec_map["asdf"] = Vec{0.1f, 0.2f, 0.3f};
        vec_map["gfds"] = Vec{0.2f, 0.4f, 0.6f};
        vec_map["fdsa"] = Vec{0.3f, 0.5f, 0.7f};
        return vec_map;
    }
};

template <>
struct TestData<std::map<uint32_t, Vec>> : EqualHelper
{
    static std::map<uint32_t, Vec> init()
    {
        std::map<uint32_t, Vec> vec_map;
        vec_map[0] = Vec{0.1f, 0.2f, 0.3f};
        vec_map[1] = Vec{0.2f, 0.4f, 0.6f};
        vec_map[5] = Vec{0.3f, 0.5f, 0.7f};
        return vec_map;
    }
};

template <>
struct TestData<mo::KVP<std::string, Vec>> : EqualHelper
{
    static mo::KVP<std::string, Vec> init()
    {
        mo::KVP<std::string, Vec> kvp;
        kvp.key = "asdf";
        kvp.value = Vec{0.f, 1.f, 2.f};
        return kvp;
    }
};

template <>
struct TestData<mo::KVP<uint32_t, Vec>> : EqualHelper
{
    static mo::KVP<uint32_t, Vec> init()
    {
        mo::KVP<uint32_t, Vec> kvp;
        kvp.key = 10;
        kvp.value = Vec{0.f, 1.f, 2.f};
        return kvp;
    }
};

template <>
struct TestData<std::vector<std::shared_ptr<Vec>>>
{
    static std::vector<std::shared_ptr<Vec>> init()
    {
        std::vector<std::shared_ptr<Vec>> out;
        out.push_back(std::make_shared<Vec>(Vec{0, 1, 2}));
        out.push_back(std::make_shared<Vec>(Vec{2, 3, 4}));
        out.push_back(std::make_shared<Vec>(Vec{5, 6, 7}));
        out.push_back(out[0]);
        return out;
    }

    static testing::AssertionResult Compare(const char* lhs_expression,
                                            const char* rhs_expression,
                                            const std::vector<std::shared_ptr<Vec>>& lhs,
                                            const std::vector<std::shared_ptr<Vec>>& rhs)
    {
        if (lhs.size() != rhs.size())
        {
            return testing::internal::EqFailure(
                "lhs.size()",
                "rhs.size()",
                testing::internal::FormatForComparisonFailureMessage(lhs.size(), rhs.size()),
                testing::internal::FormatForComparisonFailureMessage(rhs.size(), lhs.size()),
                false);
        }
        if (lhs.size() != 4)
        {
            return testing::internal::EqFailure("lhs.size()",
                                                "4",
                                                testing::internal::FormatForComparisonFailureMessage(lhs.size(), 4),
                                                testing::internal::FormatForComparisonFailureMessage(4, lhs.size()),
                                                false);
        }
        for (size_t i = 0; i < 3; ++i)
        {
            if (lhs[i] != nullptr && rhs[i] == nullptr)
            {
                return testing::internal::EqFailure(
                    "lhs[i]",
                    "rhs[i]",
                    testing::internal::FormatForComparisonFailureMessage(lhs[i], rhs[i]),
                    testing::internal::FormatForComparisonFailureMessage(rhs[i], lhs[i]),
                    false);
            }

            if (*lhs[i] != *rhs[i])
            {
                return testing::internal::EqFailure(
                    "*lhs[i]",
                    "*rhs[i]",
                    testing::internal::FormatForComparisonFailureMessage(*lhs[i], *rhs[i]),
                    testing::internal::FormatForComparisonFailureMessage(*rhs[i], *lhs[i]),
                    false);
            }
        }
        if (rhs[0] != rhs[3])
        {
            return testing::internal::EqFailure("rhs[0]",
                                                "rhs[3]",
                                                testing::internal::FormatForComparisonFailureMessage(rhs[0], rhs[3]),
                                                testing::internal::FormatForComparisonFailureMessage(rhs[3], rhs[0]),
                                                false);
        }
        return testing::AssertionSuccess();
    }
};
#ifdef MO_HAVE_OPENCV
#include <opencv2/core/types.hpp>
TEST_DATA(cv::Rect2f, {0.1f, 0.2f, 0.3f, 0.4f});
TEST_DATA(cv::Rect, {0, 2, 4, 5});
TEST_DATA(cv::Point2f, {0, 1});
TEST_DATA(cv::Matx33f, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
TEST_DATA(cv::Matx22f, {0.0f, 1.0f, 2.0f, 3.0f});
TEST_DATA(cv::Matx44f,
          {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F});
TEST_DATA(cv::Vec2b, {0, 1});
TEST_DATA(cv::Vec2i, {0, 3});
/*template <>
struct TestData<cv::Mat>
{
    static cv::Mat init()
    {
        return cv::Mat(std::vector<float>{0.0F,1.1F,2.2F,3.3F,4.4F,5.5F});
    }

    static testing::AssertionResult Compare(const char* lhs_expression, const char* rhs_expression, const cv::Mat& lhs,
const cv::Mat rhs)
    {
        if(lhs.size() != rhs.size())
        {
            return testing::internal::EqFailure("lhs.size()", "rhs.size()",
            testing::internal::FormatForComparisonFailureMessage(lhs.size(), ))
        }
    }
};*/
#endif

using String = std::string;
using VecOfStrings = std::vector<std::string>;
using RuntimeReflectionTypeTest = ::testing::Types<int,
                                                   float,
                                                   double,
                                                   String,
                                                   VecOfStrings,
#ifdef MO_HAVE_OPENCV
                                                   cv::Rect2f,
                                                   cv::Rect,
                                                   cv::Point2f,
                                                   cv::Matx33f,
                                                   cv::Matx22f,
                                                   cv::Matx44f,
                                                   // cv::Mat,
                                                   cv::Vec2b,
                                                   cv::Vec2i,
#endif
                                                   TestPodStruct,
                                                   std::vector<Vec>,
                                                   VecOwner,
                                                   std::shared_ptr<Vec>,
                                                   std::map<std::string, Vec>,
                                                   mo::KVP<std::string, Vec>,
                                                   mo::KVP<uint32_t, Vec>,
                                                   std::map<uint32_t, Vec>,
                                                   std::vector<std::shared_ptr<Vec>>>;

template <class Tester>
void testTypes(Tester& tester)
{
    // Pod datatypes
    {
        int x = 10;
        tester.test(x);

        float y = 13.2f;
        tester.test(y);

        double z = 3.14159;
        tester.test(z);

#ifdef MO_HAVE_OPENCV
        cv::Rect2f rectf(0.1f, 0.2f, 0.3f, 0.4f);
        tester.test(rectf);

        cv::Rect rect(0, 2, 3, 4);
        tester.test(rect);

        cv::Point2f point(0, 1);
        tester.test(point);
        {
            cv::Matx33f matx{0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
            tester.test(matx);
        }
        {
            cv::Matx22f matx{0.0F, 1.0F, 2.0F, 3.0F};
            tester.test(matx);
        }

        {
            cv::Matx44f matx{
                0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F};
            tester.test(matx);
        }

        {
            cv::Vec2b vec{0, 1};
            tester.test(vec);
        }
        {
            cv::Vec2i vec{0, 3};
            tester.test(vec);
        }
#endif

        TestPodStruct test_struct;
        test_struct.x = 0;
        test_struct.y = 1;
        test_struct.z = 2;
        test_struct.id = 10;
        tester.test(test_struct);
    }
    {
        std::vector<Vec> vec;
        vec.push_back({0, 1, 2});
        vec.push_back({3, 4, 5});
        vec.push_back({6, 7, 8});
        vec.push_back({9, 10, 11});
        tester.test(vec);
    }
    {
        VecOwner owner;
        owner.my_vecs.push_back({0, 1, 2});
        owner.my_vecs.push_back({3, 4, 5});
        owner.my_vecs.push_back({6, 7, 8});
        owner.my_vecs.push_back({9, 10, 11});
        tester.test(owner);
    }
    {
        std::shared_ptr<Vec> vec_ptr = std::make_shared<Vec>();
        vec_ptr->x = 1.0;
        vec_ptr->y = 2.0;
        vec_ptr->z = 3.0;
        tester.test(vec_ptr);
    }
    {
        std::map<std::string, Vec> vec_map;
        vec_map["asdf"] = Vec{0.1f, 0.2f, 0.3f};
        vec_map["gfds"] = Vec{0.2f, 0.4f, 0.6f};
        vec_map["fdsa"] = Vec{0.3f, 0.5f, 0.7f};
        tester.test(vec_map);
    }

    {
        mo::KVP<uint32_t, Vec> kvp;
        kvp.key = 10;
        kvp.value = Vec{0.f, 1.f, 2.f};
        tester.test(kvp);
    }

    {
        std::map<uint32_t, Vec> vec_map;
        vec_map[0] = Vec{0.1f, 0.2f, 0.3f};
        vec_map[1] = Vec{0.2f, 0.4f, 0.6f};
        vec_map[5] = Vec{0.3f, 0.5f, 0.7f};
        tester.test(vec_map);
    }
}
