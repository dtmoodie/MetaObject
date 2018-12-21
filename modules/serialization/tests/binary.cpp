#include <MetaObject/serialization/BinaryReader.hpp>
#include <MetaObject/serialization/BinaryWriter.hpp>
#include <cereal/archives/binary.hpp>

#include "common.hpp"
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
            mo::BinaryWriter bar(ofs);
            mo::IWriteVisitor& visitor = bar;
            T tmp = data;
            visitor(&tmp, "value0");
            bar2(tmp);
        }
        {
            std::ifstream ifs("test.bin", std::ios::binary | std::ios::in);
            mo::BinaryReader bar(ifs);
            mo::IReadVisitor& visitor = bar;

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
            mo::BinaryReader bar(ifs);
            mo::IReadVisitor& visitor = bar;

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
            mo::BinaryWriter bar(ofs);
            mo::IWriteVisitor& visitor = bar;
            std::shared_ptr<T> tmp = data;
            visitor(&tmp, "value0");
            visitor(&tmp, "value1");
        }
        {
            std::ifstream ifs("test.bin", std::ios::binary | std::ios::in);
            mo::BinaryReader bar(ifs);
            mo::IReadVisitor& visitor = bar;

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
