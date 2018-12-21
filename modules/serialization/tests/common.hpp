#pragma once
#include <MetaObject/visitation/IDynamicVisitor.hpp>
#include <MetaObject/visitation/visitor_traits/array_adapter.hpp>
#include <MetaObject/visitation/visitor_traits/vector.hpp>
#include <MetaObject/visitation/visitor_traits/memory.hpp>
#include <MetaObject/visitation/visitor_traits/map.hpp>

#include <MetaObject/types/opencv.hpp>
#include <MetaObject/serialization/cereal_map.hpp>

#include <ct/reflect.hpp>
#include <ct/reflect/cerealize.hpp>
#include <ct/reflect/compare.hpp>
#include <ct/reflect/print-container-inl.hpp>

#include <opencv2/core.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/map.hpp>

#include <iostream>
#include <memory>

#include <boost/test/auto_unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

namespace mo
{

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
}

struct DebugEqual
{
    template <class T>
    bool test(const char* name, const T& lhs, const T& rhs) const
    {
        BOOST_REQUIRE_EQUAL(lhs, rhs);
        return true;
    }

    template <class T>
    bool test(const char* name, const std::shared_ptr<T>& lhs, const std::shared_ptr<T>& rhs) const
    {
        BOOST_REQUIRE_EQUAL(lhs, rhs);
        return true;
    }

    template <class T>
    bool test(const T& lhs, const T& rhs) const
    {
        BOOST_REQUIRE_EQUAL(lhs, rhs);
        return true;
    }

    template <class T>
    bool test(const std::shared_ptr<T>& lhs, const std::shared_ptr<T>& rhs) const
    {
        BOOST_REQUIRE_EQUAL(lhs, rhs);
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
    template<class K, class V>
    ostream& operator <<(ostream& os, const map<K, V>& m)
    {
        ct::printStruct(os, m);
        return os;
    }
}
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
        PUBLIC_ACCESS(my_vecs);
    REFLECT_END;
}

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

        cv::Rect2f rectf(0.1f, 0.2f, 0.3f, 0.4f);
        tester.test(rectf);

        cv::Rect rect(0, 2, 3, 4);
        tester.test(rect);

        cv::Point2f point(0, 1);
        tester.test(point);

        cv::Matx33f matx{0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
        tester.test(matx);

        TestPodStruct test_struct;
        test_struct.x = 0;
        test_struct.y = 1;
        test_struct.z = 2;
        test_struct.id = 10;
        tester.test(test_struct);
    }
    {
        std::vector<Vec> vec;
        vec.push_back({0,1,2});
        vec.push_back({3,4,5});
        vec.push_back({6,7,8});
        vec.push_back({9,10,11});
        tester.test(vec);
    }
    {
        VecOwner owner;
        owner.my_vecs.push_back({0,1,2});
        owner.my_vecs.push_back({3,4,5});
        owner.my_vecs.push_back({6,7,8});
        owner.my_vecs.push_back({9,10,11});
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
}
