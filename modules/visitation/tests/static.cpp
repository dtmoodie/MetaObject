#include "common.hpp"

#include <MetaObject/visitation.hpp>
#include <MetaObject/visitation/HashVisitor.hpp>

struct Tester
{
    template<class T>
    void test(const T& type)
    {
        mo::HashVisitor hasher;

        auto trait = mo::makeTraits<T>(static_cast<T*>(nullptr));

        hasher.generateObjecthash(&trait);
    }
};

BOOST_AUTO_TEST_CASE(Hash)
{
    Tester tester;
    testTypes(tester);
}
