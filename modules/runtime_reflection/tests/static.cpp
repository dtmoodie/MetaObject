#include "common.hpp"
#include <MetaObject/runtime_reflection/visitor_traits/array_adapter.hpp>

#include <MetaObject/runtime_reflection.hpp>
#include <MetaObject/runtime_reflection/HashVisitor.hpp>

#include "gtest/gtest.h"

namespace
{
    struct Tester
    {
        template <class T>
        typename std::enable_if<!std::is_arithmetic<T>::value>::type test(const T&)
        {
            mo::HashVisitor hasher;

            auto trait = mo::makeTraits<T>(static_cast<T*>(nullptr));
            using base = typename decltype(trait)::base;
            const auto hash = hasher.generateObjecthash(static_cast<base*>(&trait));

            m_hashes[mo::TypeInfo::create<T>().name()] = hash;

            ASSERT_EQ(std::count(m_hash_list.begin(), m_hash_list.end(), hash), 0);
            m_hash_list.push_back(hash);
        }

        template <class T>
        typename std::enable_if<std::is_arithmetic<T>::value>::type test(const T&)
        {
        }

        std::map<std::string, size_t> m_hashes;
        std::vector<size_t> m_hash_list;
    };

    struct TrivialSerializableTester
    {

        template <class T>
        typename std::enable_if<std::is_base_of<mo::IStructTraits, mo::TTraits<T>>::value &&
                                !std::is_base_of<mo::IContainerTraits, mo::TTraits<T>>::value &&
                                !std::is_base_of<mo::IPtrTraits, mo::TTraits<T>>::value>::type
        testImpl(const T& data)
        {
            auto trait = mo::makeTraits(&data);
            if (trait.triviallySerializable())
            {
                T tmp;
                // We know that this is going to be a warning,t he point of this function is to test to make sure the
                // dynamic value of trait.triviallySerializable() only lets us enter this if statement if trivially
                // serializable is
                std::memcpy(&tmp, &data, sizeof(T));
                if (!ct::compare(tmp, data, DebugEqual()))
                {
                }
            }
        }

        template <class T>
        typename std::enable_if<std::is_base_of<mo::IContainerTraits, mo::TTraits<T>>::value>::type testImpl(const T&)
        {
        }

        template <class T>
        typename std::enable_if<!std::is_base_of<mo::IStructTraits, mo::TTraits<T>>::value>::type testImpl(const T&)
        {
        }

        template <class T>
        typename std::enable_if<std::is_base_of<mo::IPtrTraits, mo::TTraits<T>>::value>::type testImpl(const T&)
        {
        }

        template <class T>
        typename std::enable_if<!std::is_arithmetic<T>::value>::type test(const T& data)
        {
            testImpl(data);
        }

        template <class T>
        typename std::enable_if<std::is_arithmetic<T>::value>::type test(const T&)
        {
        }
    };
} // namespace

TEST(static_reflection, hash)
{
    Tester tester;
    testTypes(tester);
}

struct VecA
{
    float x, y, z;
};

struct VecB
{
    float x, y, z;
};

struct VecC
{
    float a, b, c;
};

struct Quat
{
    float x, y, z, w;
};

namespace ct
{

    REFLECT_BEGIN(VecA)
        PUBLIC_ACCESS(x)
        PUBLIC_ACCESS(y)
        PUBLIC_ACCESS(z)
    REFLECT_END;

    REFLECT_BEGIN(VecB)
        PUBLIC_ACCESS(x)
        PUBLIC_ACCESS(y)
        PUBLIC_ACCESS(z)
    REFLECT_END;

    REFLECT_BEGIN(VecC)
        PUBLIC_ACCESS(a)
        PUBLIC_ACCESS(b)
        PUBLIC_ACCESS(c)
    REFLECT_END;

    REFLECT_BEGIN(Quat)
        PUBLIC_ACCESS(x)
        PUBLIC_ACCESS(y)
        PUBLIC_ACCESS(z)
        PUBLIC_ACCESS(w)
    REFLECT_END;

} // namespace ct

TEST(static_reflection, DetectSimilarity)
{
    mo::TTraits<std::array<float, 4>, 4, void> test;
    {
        mo::HashVisitor hasher(true, false);
        const auto trait_a = mo::makeTraits(static_cast<VecA*>(nullptr));
        const auto hash_a = hasher.generateObjecthash(&trait_a, "");

        const auto trait_b = mo::makeTraits(static_cast<VecB*>(nullptr));
        const auto hash_b = hasher.generateObjecthash(&trait_b, "");

        const auto trait_c = mo::makeTraits(static_cast<VecC*>(nullptr));
        const auto hash_c = hasher.generateObjecthash(&trait_c, "");

        const auto trait_q = mo::makeTraits(static_cast<Quat*>(nullptr));
        const auto hash_q = hasher.generateObjecthash(&trait_q, "");

        ASSERT_EQ(hash_a, hash_b);
        ASSERT_NE(hash_a, hash_c);
        ASSERT_NE(hash_a, hash_q);
        ASSERT_NE(hash_c, hash_q);
    }

    {
        mo::HashVisitor hasher(false, false);
        const auto trait_a = mo::makeTraits(static_cast<VecA*>(nullptr));
        const auto hash_a = hasher.generateObjecthash(&trait_a, "");

        const auto trait_b = mo::makeTraits(static_cast<VecB*>(nullptr));
        const auto hash_b = hasher.generateObjecthash(&trait_b, "");

        const auto trait_c = mo::makeTraits(static_cast<VecC*>(nullptr));
        const auto hash_c = hasher.generateObjecthash(&trait_c, "");

        const auto trait_q = mo::makeTraits(static_cast<Quat*>(nullptr));
        const auto hash_q = hasher.generateObjecthash(&trait_q, "");

        ASSERT_EQ(hash_a, hash_b);
        ASSERT_EQ(hash_a, hash_c);
        ASSERT_NE(hash_a, hash_q);
        ASSERT_NE(hash_c, hash_q);
    }
}

TEST(static_reflection, TrivialSerializability)
{
    TrivialSerializableTester tester;
    testTypes(tester);
}
