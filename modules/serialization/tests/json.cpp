#include "common.hpp"
#include <MetaObject/serialization/JSONPrinter.hpp>

struct TestJson
{
    template <class T>
    void test(const T& data)
    {
        T tmp = data;
        std::stringstream ss;
        {
            mo::JSONWriter printer(ss);
            mo::IWriteVisitor& visitor = printer;
            visitor(&tmp);
        }
        ss.seekg(std::ios::beg);
        std::cout << "------------------------------\nDynamic\n";
        std::cout << ss.str() << std::endl;
        std::cout << "------------------------------\nStatic\n";
        {
            cereal::JSONOutputArchive ar(std::cout);
            ar(tmp);
        }

        {
            cereal::JSONInputArchive ar(ss);
            T tmp1;
            ar(tmp1);
            if (!ct::compare(tmp, tmp1, DebugEqual()))
            {
                std::cout << "Failed to load from json " << ct::Reflect<T>::getName() << " correctly";
                throw std::runtime_error("Static Json deserialization failed");
            }
        }

        {
            std::stringstream ss1(ss.str());
            // TODO dynamic json deserialization
            mo::JSONReader reader(ss1);
            mo::IReadVisitor& visitor = reader;
            T tmp1;
            visitor(&tmp1);
            if (!ct::compare(tmp, tmp1, DebugEqual()))
            {
                std::cout << "Failed to load from json " << ct::Reflect<T>::getName() << " correctly";
                throw std::runtime_error("Dynamic Json deserialization failed");
            }
        }
    }

    template <class T>
    void test(const std::shared_ptr<T>& data)
    {
        std::shared_ptr<T> tmp = data;
        std::stringstream ss;
        {
            mo::JSONWriter printer(ss);
            mo::IWriteVisitor& visitor = printer;
            visitor(&tmp);
        }
        ss.seekg(std::ios::beg);
        std::cout << "------------------------------\nDynamic\n";
        std::cout << ss.str() << std::endl;
        std::cout << "------------------------------\nStatic\n";
        {
            cereal::JSONOutputArchive ar(std::cout);
            ar(tmp);
        }
    }
};

BOOST_AUTO_TEST_CASE(JsonSerialization)
{
    TestJson tester;

    testTypes(tester);
}
