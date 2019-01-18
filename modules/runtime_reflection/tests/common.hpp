#pragma once
#ifdef HAVE_OPENCV
#include <MetaObject/types/opencv.hpp>
#include <opencv2/core.hpp>
#endif

#include <MetaObject/runtime_reflection.hpp>

#include <MetaObject/runtime_reflection/visitor_traits/array_adapter.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/map.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/memory.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/vector.hpp>

#include <ct/reflect.hpp>
#include <ct/reflect/cerealize.hpp>
#include <ct/reflect/compare.hpp>
#include <ct/reflect/print-container-inl.hpp>

#include <cereal/types/map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>

#include <iostream>
#include <memory>

#include <boost/test/auto_unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

namespace mo
{
    struct PrintVisitor : public mo::StaticVisitor
    {
        int indent = 2;

        void visit(const mo::ITraits* trait, const std::string& name, const size_t cnt = 1) override
        {
            for (int i = 0; i < indent; ++i)
            {
                std::cout << ' ';
            }
            indent += 2;
            std::cout << name << ": " << trait->getName() << std::endl;
            m_path.push_back(trait->getName());
            trait->visit(this);
            indent -= 2;
            m_path.pop_back();
        }

        void implDyn(const mo::TypeInfo type, const std::string& name, const size_t cnt) override
        {
            for (int i = 0; i < indent; ++i)
            {
                std::cout << ' ';
            }
            std::cout << name << ": " << mo::TypeTable::instance().typeToName(type) << std::endl;
            m_items.push_back(getName() + "." + name);
        }

        std::string getName()
        {
            std::stringstream ss;
            for(const std::string& path : m_path)
            {
                ss << path;
                ss << '.';
            }
            return std::move(ss).str();
        }
        std::vector<std::string> m_path;
        std::vector<std::string> m_items;
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

    template <class T, size_t N>
    bool operator==(const mo::ArrayAdapter<T, N>& lhs, const mo::ArrayAdapter<T, N>& rhs)
    {
        for (size_t i = 0; i < N; ++i)
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
    template <class K, class V>
    ostream& operator<<(ostream& os, const map<K, V>& m)
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
        PUBLIC_ACCESS(my_vecs)
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

#ifdef HAVE_OPENCV
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
