#include "common.hpp"

#include <MetaObject/core/detail/Time.hpp>
#include <MetaObject/serialization/BinaryLoader.hpp>
#include <MetaObject/serialization/BinarySaver.hpp>
#include <MetaObject/types/cereal_map.hpp>

#include <cereal/archives/binary.hpp>

#include <fstream>
#include <iostream>

struct TestBinary
{
    template <class T>
    void test(const T& data)
    {
        {
            std::ofstream ofs("test.bin", std::ios::binary | std::ios::out);
            std::ofstream ofs2("test_cereal.bin", std::ios::binary | std::ios::out);
            cereal::BinaryOutputArchive bar2(ofs2);
            mo::BinarySaver bar(ofs);
            mo::ISaveVisitor& visitor = bar;
            T tmp = data;
            visitor(&tmp, "value0");
            bar2(tmp);
        }
        {
            std::ifstream ifs("test.bin", std::ios::binary | std::ios::in);
            mo::BinaryLoader bar(ifs);
            mo::ILoadVisitor& visitor = bar;

            T tmp;
            visitor(&tmp, "value0");
            if (!ct::compare(tmp, data, DebugEqual()))
            {
                std::cout << "Failed to serialize " << ct::Reflect<T>::getName() << " correctly";
                throw std::runtime_error("Serialization failed");
            }
        }

        {
            std::ifstream ifs("test_cereal.bin", std::ios::binary | std::ios::in);
            mo::BinaryLoader bar(ifs);
            mo::ILoadVisitor& visitor = bar;

            T tmp;
            visitor(&tmp, "value0");
            if (!ct::compare(tmp, data, DebugEqual()))
            {
                std::cout << "Failed to serialize " << ct::Reflect<T>::getName() << " correctly";
                throw std::runtime_error("Serialization failed");
            }
        }

        {
            std::ifstream ifs("test.bin", std::ios::binary | std::ios::in);
            cereal::BinaryInputArchive bar(ifs);

            T tmp;
            bar(tmp);
            if (!ct::compare(tmp, data, DebugEqual()))
            {
                std::cout << "Failed to serialize " << ct::Reflect<T>::getName() << " correctly";
                throw std::runtime_error("Serialization failed");
            }
        }
    }

    template <class T>
    void test(const std::shared_ptr<T>& data)
    {
        {
            std::ofstream ofs("test.bin", std::ios::binary | std::ios::out);
            mo::BinarySaver bar(ofs);
            mo::ISaveVisitor& visitor = bar;
            std::shared_ptr<T> tmp = data;
            visitor(&tmp, "value0");
            visitor(&tmp, "value1");
        }
        {
            std::ifstream ifs("test.bin", std::ios::binary | std::ios::in);
            mo::BinaryLoader bar(ifs);
            mo::ILoadVisitor& visitor = bar;

            std::shared_ptr<T> tmp;
            std::shared_ptr<T> tmp2;
            visitor(&tmp, "value0");
            visitor(&tmp2, "value1");
            if (!ct::compare(*tmp, *data, DebugEqual()))
            {
                std::cout << "Failed to serialize " << ct::Reflect<T>::getName() << " correctly";
                throw std::runtime_error("Serialization failed");
            }
            if (tmp.get() != tmp2.get())
            {
                std::cout << "Failed to reshare ownership of two smart pointers" << std::endl;
                throw std::runtime_error("Serialization failed");
            }
        }
    }
};

BOOST_AUTO_TEST_CASE(BinarySerialization)
{
    TestBinary tester;

    testTypes(tester);
}

template <class T>
void testBinarySpeed(size_t count)
{
    std::vector<T> vec(count);
    std::cout << "------ " << mo::TypeInfo(typeid(T)).name() << " ------ " << std::endl;
    {
        mo::Time start = mo::Time::now();
        std::ofstream ofs("test.bin", std::ios::binary | std::ios::out);
        mo::BinarySaver bar(ofs);
        mo::ISaveVisitor& visitor = bar;
        visitor(&vec);
        mo::Time end = mo::Time::now();
        std::cout << "Dynamic binary write time: ";
        mo::Time delta(end - start);
        delta.print(std::cout, false, false, true, true, true);
        std::cout << std::endl;
    }
    {
        mo::Time start = mo::Time::now();
        std::ifstream ifs("test.bin", std::ios::binary | std::ios::in);
        mo::BinaryLoader bar(ifs);
        mo::ILoadVisitor& visitor = bar;
        std::vector<T> readin;
        visitor(&readin);
        mo::Time end = mo::Time::now();
        std::cout << "Dynamic binary read time: ";
        mo::Time delta(end - start);
        delta.print(std::cout, false, false, true, true, true);
        std::cout << std::endl;
    }

    {
        mo::Time start = mo::Time::now();
        std::ofstream ofs("test.bin", std::ios::binary | std::ios::out);
        cereal::BinaryOutputArchive bar(ofs);
        bar(vec);
        mo::Time end = mo::Time::now();
        std::cout << "Cereal binary write time: ";
        mo::Time delta(end - start);
        delta.print(std::cout, false, false, true, true, true);
        std::cout << std::endl;
    }

    {
        mo::Time start = mo::Time::now();
        std::ifstream ofs("test.bin", std::ios::binary | std::ios::in);
        cereal::BinaryInputArchive bar(ofs);
        std::vector<T> readin;
        bar(readin);
        mo::Time end = mo::Time::now();

        std::cout << "Cereal binary read time: ";
        mo::Time delta(end - start);
        delta.print(std::cout, false, false, true, true, true);
        std::cout << std::endl;
    }
}

BOOST_AUTO_TEST_CASE(BinarySerializationPerformance)
{
    testBinarySpeed<float>(1e9);
    testBinarySpeed<TestPodStruct>(1e7);
}
