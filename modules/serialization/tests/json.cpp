#include <common.hpp>
#include <MetaObject/types/cereal_map.hpp>

#include <MetaObject/serialization/JSONPrinter.hpp>

struct TestJson
{
    template <class T>
    void test(const T& data)
    {
        T tmp = data;
        std::stringstream dyn, stat;
        {
            mo::JSONSaver printer(dyn);
            mo::ISaveVisitor& visitor = printer;
            visitor(&tmp);
            dyn.seekg(std::ios::beg);
        }

        {
            cereal::JSONOutputArchive ar(stat);
            ar(tmp);
        }

        std::cout << "------------------------------\nDynamic\n";
        std::cout << dyn.str() << std::endl;
        std::cout << "------------------------------\nStatic\n";
        std::cout << stat.str() << std::endl;

        {
            cereal::JSONInputArchive ar(dyn);
            T tmp1;
            ar(tmp1);
            if (!ct::compare(tmp, tmp1, DebugEqual()))
            {
                std::cout << "Failed to load from json " << ct::Reflect<T>::getName() << " correctly";
                throw std::runtime_error("Static Json deserialization failed");
            }
        }

        {
            std::stringstream ss1(dyn.str());
            mo::JSONLoader reader(ss1);
            mo::ILoadVisitor& visitor = reader;
            T tmp1;
            visitor(&tmp1);
            if (!ct::compare(tmp, tmp1, DebugEqual()))
            {
                std::cout << "Failed to load from json " << ct::Reflect<T>::getName() << " correctly";
                throw std::runtime_error("Dynamic Json deserialization of dynamically built serialization failed");
            }
        }

        {
            std::stringstream ss1(stat.str());
            mo::JSONLoader reader(ss1);
            mo::ILoadVisitor& visitor = reader;
            T tmp1;
            visitor(&tmp1);
            if (!ct::compare(tmp, tmp1, DebugEqual()))
            {
                std::cout << "Failed to load from json " << ct::Reflect<T>::getName() << " correctly";
                throw std::runtime_error("Dynamic Json deserialization of statically built serialization failed");
            }
        }
    }

    template <class T>
    void test(const std::shared_ptr<T>& data)
    {
        std::shared_ptr<T> tmp = data;
        std::stringstream ss;
        {
            mo::JSONSaver printer(ss);
            mo::ISaveVisitor& visitor = printer;
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
