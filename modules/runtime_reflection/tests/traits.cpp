#include "common.hpp"
#include <MetaObject/runtime_reflection/TraitRegistry.hpp>

BOOST_AUTO_TEST_CASE(TraitRegistry)
{
    using namespace mo;
    const auto& known_traits = mo::TraitRegistry::instance().getTraits();
    PrintVisitor visitor;
    for (const auto& trait : known_traits)
    {
        std::cout << trait.first.name() << std::endl;
        trait.second->visit(&visitor);
    }
}
