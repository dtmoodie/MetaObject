#include "common.hpp"
#include <MetaObject/visitation/TraitRegistry.hpp>



BOOST_AUTO_TEST_CASE(TraitRegistry)
{
    const auto& known_traits = mo::TraitRegistry::instance();
}
