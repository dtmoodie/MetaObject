#include <../tests/reflect/Data.hpp>
#include <../tests/reflect/Reflect.hpp>
#include <../tests/reflect/common.hpp>

#include <MetaObject/serialization/BinaryReader.hpp>

#include <MetaObject/serialization/BinaryWriter.hpp>
#include <MetaObject/serialization/JSONPrinter.hpp>

#include <MetaObject/visitation/visitor_traits/map.hpp>
#include <MetaObject/visitation/visitor_traits/memory.hpp>
#include <MetaObject/visitation/visitor_traits/string.hpp>
#include <MetaObject/visitation/visitor_traits/vector.hpp>

#include <ct/reflect.hpp>
#include <ct/reflect/cerealize.hpp>
#include <ct/reflect/compare-inl.hpp>
#include <ct/reflect/compare.hpp>
#include <ct/reflect/print.hpp>

#include <ct/reflect/compare-container-inl.hpp>

#include <cereal/archives/json.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>

#include <cassert>
#include <fstream>
#include <map>
#include <type_traits>

#include <boost/test/auto_unit_test.hpp>
#include <boost/test/test_tools.hpp>

namespace cereal
{
    //! Saving for std::map<std::string, std::string> for text based archives
    // Note that this shows off some internal cereal traits such as EnableIf,
    // which will only allow this template to be instantiated if its predicates
    // are true
    template <class Archive,
              class T,
              class C,
              class A,
              traits::EnableIf<traits::is_text_archive<Archive>::value> = traits::sfinae>
    inline void save(Archive& ar, std::map<std::string, T, C, A> const& map)
    {
        for (const auto& i : map)
            ar(cereal::make_nvp(i.first, i.second));
    }

    //! Loading for std::map<std::string, std::string> for text based archives
    template <class Archive,
              class T,
              class C,
              class A,
              traits::EnableIf<traits::is_text_archive<Archive>::value> = traits::sfinae>
    inline void load(Archive& ar, std::map<std::string, T, C, A>& map)
    {
        map.clear();

        auto hint = map.begin();
        while (true)
        {
            const auto namePtr = ar.getNodeName();

            if (!namePtr)
                break;

            std::string key = namePtr;
            T value;
            ar(value);
            hint = map.emplace_hint(hint, std::move(key), std::move(value));
        }
    }
} // namespace cereal

int main()
{
    {
        using Accessor_t = decltype(ct::Reflect<WeirdWeakOwnerShip>::getAccessor(ct::Indexer<1>{}));
        using Get_t = Accessor_t::RetType;
        static_assert(std::is_same<const std::vector<PointerOwner>&, Get_t>::value, "asdf");
    }
    /*TestBinary tester;
    testTypes(tester);

    TestJson test_json;
    testTypes(test_json);

    std::cout << std::endl;
    std::shared_ptr<ReflectedStruct> shared_ptr = std::make_shared<ReflectedStruct>();

    {
        mo::JSONWriter writer(std::cout);
        mo::IWriteVisitor& visitor = writer;
        visitor(&shared_ptr);
        visitor(&shared_ptr);
    }
    std::cout << std::endl;*/
}
