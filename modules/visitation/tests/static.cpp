#include "common.hpp"

#include <MetaObject/visitation.hpp>
#include <MetaObject/visitation/HashVisitor.hpp>

namespace
{
    struct Tester
    {
        template<class T>
        typename std::enable_if<!std::is_arithmetic<T>::value>::type test(const T&)
        {
            mo::HashVisitor hasher;

            auto trait = mo::makeTraits<T>(static_cast<T*>(nullptr));

            const auto hash = hasher.generateObjecthash(&trait);

            m_hashes[mo::TypeInfo(typeid(T)).name()] = hash;

            BOOST_REQUIRE_EQUAL(std::count(m_hash_list.begin(),m_hash_list.end(), hash), 0);
            m_hash_list.push_back(hash);
        }

        template<class T>
        typename std::enable_if<std::is_arithmetic<T>::value>::type test(const T&)
        {

        }

        std::map<std::string, size_t> m_hashes;
        std::vector<size_t> m_hash_list;
    };
}

BOOST_AUTO_TEST_CASE(Hash)
{
    Tester tester;
    testTypes(tester);
}

struct VecA
{
    float x,y,z;
};

struct VecB
{
    float x,y,z;
};

struct VecC
{
    float a,b,c;
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

}

BOOST_AUTO_TEST_CASE(DetectSimilarity)
{

    {
        mo::HashVisitor hasher(true, false);
        const auto trait_a = mo::makeTraits(static_cast<VecA*>(nullptr));
        const auto hash_a = hasher.generateObjecthash(&trait_a);

        const auto trait_b = mo::makeTraits(static_cast<VecB*>(nullptr));
        const auto hash_b = hasher.generateObjecthash(&trait_b);

        const auto trait_c = mo::makeTraits(static_cast<VecC*>(nullptr));
        const auto hash_c = hasher.generateObjecthash(&trait_c);

        const auto trait_q = mo::makeTraits(static_cast<Quat*>(nullptr));
        const auto hash_q = hasher.generateObjecthash(&trait_q);

        BOOST_REQUIRE_EQUAL(hash_a, hash_b);
        BOOST_REQUIRE_NE(hash_a, hash_c);
        BOOST_REQUIRE_NE(hash_a, hash_q);
        BOOST_REQUIRE_NE(hash_c, hash_q);
    }

    {
        mo::HashVisitor hasher(false, false);
        const auto trait_a = mo::makeTraits(static_cast<VecA*>(nullptr));
        const auto hash_a = hasher.generateObjecthash(&trait_a);

        const auto trait_b = mo::makeTraits(static_cast<VecB*>(nullptr));
        const auto hash_b = hasher.generateObjecthash(&trait_b);

        const auto trait_c = mo::makeTraits(static_cast<VecC*>(nullptr));
        const auto hash_c = hasher.generateObjecthash(&trait_c);

        const auto trait_q = mo::makeTraits(static_cast<Quat*>(nullptr));
        const auto hash_q = hasher.generateObjecthash(&trait_q);

        BOOST_REQUIRE_EQUAL(hash_a, hash_b);
        BOOST_REQUIRE_EQUAL(hash_a, hash_c);
        BOOST_REQUIRE_NE(hash_a, hash_q);
        BOOST_REQUIRE_NE(hash_c, hash_q);
    }
}
