#include "common.hpp"
#include <MetaObject/runtime_reflection/TraitRegistry.hpp>

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
        trait->visit(this);
        indent -= 2;
    }

    void implDyn(const mo::TypeInfo type, const std::string& name, const size_t cnt) override
    {
        for (int i = 0; i < indent; ++i)
        {
            std::cout << ' ';
        }
        std::cout << name << ": " << mo::TypeTable::instance().typeToName(type) << std::endl;
    }
};

BOOST_AUTO_TEST_CASE(TraitRegistry)
{
    const auto& known_traits = mo::TraitRegistry::instance().getTraits();
    PrintVisitor visitor;
    for (const auto& trait : known_traits)
    {
        std::cout << trait.first.name() << std::endl;
        trait.second->visit(&visitor);
    }
}
