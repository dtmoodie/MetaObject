#include "TraitInterface.hpp"
#include "TraitRegistry.hpp"
namespace mo
{

std::string ITraits::name() const
{
    const auto t = type();
    return t.name();
}

ITraits::~ITraits() = default;


}
