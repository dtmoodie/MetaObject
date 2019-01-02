#include "common.hpp"

#include <MetaObject/visitation.hpp>
#include <MetaObject/visitation/HashVisitor.hpp>

struct Tester
{
    template<class T>
    typename std::enable_if<!std::is_arithmetic<T>::value>::type test(const T&)
    {
        mo::HashVisitor hasher;

        auto trait = mo::makeTraits<T>(static_cast<T*>(nullptr));

        const auto hash = hasher.generateObjecthash(&trait);
        m_hashes[mo::TypeInfo(typeid(T)).name()] = hash;
    }

    template<class T>
    typename std::enable_if<std::is_arithmetic<T>::value>::type test(const T&)
    {

    }

    std::map<std::string, size_t> m_hashes;
};

BOOST_AUTO_TEST_CASE(Hash)
{
    Tester tester;
    testTypes(tester);
}
