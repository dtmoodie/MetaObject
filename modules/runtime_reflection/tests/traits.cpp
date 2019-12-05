#include <ct/reflect_traits.hpp>

#include "common.hpp"
#include <MetaObject/runtime_reflection/TraitRegistry.hpp>

#include <gtest/gtest.h>
namespace
{

    // TODO improve tester to verify that all fields found in runtime reflection are also found in compile time
    // reflection

    struct ReflectionVisitor : public mo::StaticVisitor
    {
        void visit(const mo::ITraits* trait, const std::string& name, size_t cnt = 1)
        {
            auto path = makePath();
            if (!path.empty())
            {
                path = path + name;
            }
            else
            {
                path = name;
            }
            data.emplace_back(path, trait->type());
            m_path.push_back(name);
            trait->visit(*this, name);
            m_path.pop_back();
        }

        void implDyn(mo::TypeInfo type, const std::string& name, size_t cnt)
        {
            data.emplace_back(name, type);
        }

        std::string makePath()
        {
            std::stringstream ss;
            for (const auto str : m_path)
            {
                ss << str << '.';
            }
            return ss.str();
        }
        std::vector<std::pair<std::string, mo::TypeInfo>> data;
        std::vector<std::string> m_path;
    };

    struct TraitTester
    {
        std::vector<std::pair<std::string, mo::TypeInfo>> data;
        template <class T, ct::index_t I>
        void checkField(ct::Indexer<I> idx)
        {
            auto ptr = ct::Reflect<T>::getPtr(idx);
            if (ct::IsWritable<T, I>::value)
            {
                auto name = ptr.getName();
                using type = ct::decay_t<typename ct::FieldGetType<T, I>::type>;
                auto itr =
                    std::find_if(data.begin(), data.end(), [name](std::pair<std::string, mo::TypeInfo> data) -> bool {
                        return name == data.first;
                    });
                ASSERT_NE(itr, data.end())
                    << "Unable to find trait by name " << name << " for type " << ct::Reflect<T>::getName();
                EXPECT_EQ(itr->second, mo::TypeInfo::template create<type>());
            }
        }

        template <class T>
        void checkFieldRecurse(ct::Indexer<0> idx)
        {
            checkField<T>(idx);
        }

        template <class T, ct::index_t I>
        void checkFieldRecurse(const mo::ITraits&, ct::Indexer<I> idx)
        {
            checkField<T>(idx);
            checkFieldRecurse<T>(--idx);
        }

        template <class T>
        auto test(const T&) -> ct::EnableIf<ct::IsReflected<T>::value>
        {
            const auto& known_traits = mo::TraitRegistry::instance().getTraits();
            auto type = mo::TypeInfo::template create<T>();
            auto itr = known_traits.find(type);
            if (itr == known_traits.end())
            {
                std::cout << "Unable to find trait for " << type.name() << std::endl;
            }
            EXPECT_NE(itr, known_traits.end());
            ReflectionVisitor visitor;
            itr->second->visit(visitor, "");
            data = visitor.data;
            checkField<T>(ct::Reflect<T>::end());
        }

        template <class T>
        auto test(const T&) -> ct::DisableIf<ct::IsReflected<T>::value>
        {
        }
    };
} // namespace

TEST(runtime_reflection, TraitRegistry)
{
    using namespace mo;
    const auto& known_traits = mo::TraitRegistry::instance().getTraits();
    PrintVisitor visitor(std::cout);
    EXPECT_GE(known_traits.size(), 5);
    for (const auto& trait : known_traits)
    {
        std::cout << trait.first.name() << std::endl;
        trait.second->visit(visitor, "");
    }
    TraitTester tester;
    testTypes(tester);
}
