#include "common.hpp"

#include <gtest/gtest.h>

#include <MetaObject/core/detail/Time.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/map.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/pair.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/vector.hpp>
#include <MetaObject/serialization/BinaryLoader.hpp>
#include <MetaObject/serialization/BinarySaver.hpp>
#include <MetaObject/types/cereal_map.hpp>

#include <cereal/archives/binary.hpp>

#include <MetaObject/serialization/JSONPrinter.hpp>

#include <fstream>
#include <iostream>

template <class T>
struct Serialization : ::testing::Test
{

    static std::string binaryCerealStr()
    {
        std::stringstream ss;
        {
            cereal::BinaryOutputArchive bar(ss);
            auto data = TestData<T>::init();
            bar(data);
        }
        return std::move(ss).str();
    }

    static void saveBinaryCereal(std::string path)
    {
        auto data = TestData<T>::init();
        std::ofstream ofs2(path, std::ios::binary | std::ios::out);
        cereal::BinaryOutputArchive bar2(ofs2);
        bar2(data);
    }

    static void saveBinaryRuntime(std::string path)
    {
        std::ofstream ofs(path, std::ios::binary | std::ios::out);
        mo::BinarySaver bar(ofs);
        mo::ISaveVisitor& visitor = bar;
        auto data = TestData<T>::init();
        visitor(&data, "value0");
    }

    static void loadBinaryCereal(std::string path)
    {
        std::ifstream ifs(path, std::ios::binary | std::ios::in);
        cereal::BinaryInputArchive bar(ifs);
        T tmp;
        std::string error;
        try
        {
            bar(tmp);
        }
        catch (const std::exception& e)
        {
            error = e.what();
        }

        ifs.seekg(0);
        auto data = TestData<T>::init();
        EXPECT_PRED_FORMAT2(TestData<T>::Compare, data, tmp)
            << "Failed to use cereal to deserialize a binary blob from runtime reflection serialization. " << error
            << " got:\n"
            << ifs.rdbuf() << "\nexpected:\n"
            << binaryCerealStr();
    }

    static void loadBinaryRuntime(std::string path)
    {
        std::ifstream ifs(path, std::ios::binary | std::ios::in);
        mo::BinaryLoader bar(ifs);
        mo::ILoadVisitor& visitor = bar;

        T tmp;
        visitor(&tmp, "value0");
        auto data = TestData<T>::init();
        EXPECT_PRED_FORMAT2(TestData<T>::Compare, data, tmp)
            << "Failed to use runtime reflection for binary serialization";
    }

    /*void testBinary()
    {
        auto data = TestData<T>::init();
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
            EXPECT_PRED_FORMAT2(TestData<T>::Compare, data, tmp)
                << "Failed to use runtime reflection for binary serialization";
        }

        {
            std::ifstream ifs("test_cereal.bin", std::ios::binary | std::ios::in);
            mo::BinaryLoader bar(ifs);
            mo::ILoadVisitor& visitor = bar;

            T tmp;
            visitor(&tmp, "value0");
            EXPECT_PRED_FORMAT2(TestData<T>::Compare, data, tmp) << "Failed to use cereal for binary serialization";
        }

        {
            std::ifstream ifs("test.bin", std::ios::binary | std::ios::in);
            cereal::BinaryInputArchive bar(ifs);

            T tmp;
            bar(tmp);
            EXPECT_PRED_FORMAT2(TestData<T>::Compare, data, tmp)
                << "Failed to use cereal to deserialize a binary blob from runtime reflection serialization";
        }
    }*/

    void testJsonCerealSaveCerealLoad()
    {
        T data = TestData<T>::init();
        std::stringstream stat;
        {
            cereal::JSONOutputArchive ar(stat);
            ar(data);
        }

        {
            cereal::JSONInputArchive ar(stat);
            T loaded;
            ar(loaded);
            EXPECT_PRED_FORMAT2(TestData<T>::Compare, loaded, data) << "Failed to correctly deserialize \n"
                                                                    << stat.str();
        }
    }

    void testJsonCerealSaveRuntimeReflectionLoad()
    {
        T data = TestData<T>::init();
        std::stringstream stat;
        {
            cereal::JSONOutputArchive ar(stat);
            ar(data);
        }
        bool success = true;
        std::string error;
        try
        {
            std::stringstream ss1(stat.str());
            mo::JSONLoader reader(ss1);
            mo::ILoadVisitor& visitor = reader;
            T loaded;
            visitor(&loaded);
            EXPECT_PRED_FORMAT2(TestData<T>::Compare, loaded, data)
                << "Unable to use runtime reflection to load from a json created with cereal\n"
                << stat.str();
        }
        catch (const std::exception& e)
        {
            success = false;
            error = e.what();
        }
        catch (...)
        {
            success = false;
        }
        EXPECT_TRUE(success) << "Unable to use runtime reflection to load from a json created with cereal due to "
                             << error << "\n"
                             << stat.str();
    }

    void testJsonRuntimeReflectionSaveCerealLoad()
    {
        T data = TestData<T>::init();
        std::stringstream sstream;
        {
            mo::JSONSaver saver_(sstream);
            mo::ISaveVisitor& saver = saver_;
            saver(&data);
        }
        {
            bool success = true;
            std::string error;
            try
            {
                cereal::JSONInputArchive ar(sstream);
                T loaded;
                ar(loaded);
                EXPECT_PRED_FORMAT2(TestData<T>::Compare, loaded, data) << "Failed to correctly deserialize \n"
                                                                        << sstream.str();
            }
            catch (const std::exception& e)
            {
                error = e.what();
                success = false;
            }
            catch (...)
            {
                success = false;
            }
            if (!success)
            {
                std::stringstream expected_serialization;
                {
                    cereal::JSONOutputArchive ar(expected_serialization);
                    ar(data);
                }

                EXPECT_TRUE(success) << "Execption thrown when trying to deserialize. " << error << "\n"
                                     << sstream.str() << "\nexpected json\n"
                                     << expected_serialization.str();
            }
        }
    }

    void testJsonRuntimeReflectionSaveRuntimeLoad()
    {
        T data = TestData<T>::init();
        std::stringstream sstream;
        {
            mo::JSONSaver saver_(sstream);
            mo::ISaveVisitor& saver = saver_;
            saver(&data);
        }
        {
            std::stringstream ss1(sstream.str());
            mo::JSONLoader reader(ss1);
            mo::ILoadVisitor& visitor = reader;
            T loaded;
            std::string error;
            try
            {
                visitor(&loaded);
            }
            catch (const std::exception& e)
            {
                error = e.what();
            }

            EXPECT_PRED_FORMAT2(TestData<T>::Compare, loaded, data)
                << "Unable to use runtime reflection to load from a json created with cereal. " << error << "\n"
                << sstream.str();
        }
    }
};

template <class T>
struct Serialization<std::shared_ptr<T>> : ::testing::Test
{
    std::string binaryCerealStr()
    {
        std::stringstream ss;
        {
            cereal::BinaryOutputArchive bar(ss);
            auto data = TestData<std::shared_ptr<T>>::init();
            bar(data);
        }
        return std::move(ss).str();
    }

    void saveBinaryCereal(std::string path)
    {
        std::ofstream ofs(path, std::ios::binary | std::ios::out);
        cereal::BinaryOutputArchive bar(ofs);
        auto data = TestData<std::shared_ptr<T>>::init();
        bar(data);
    }

    void saveBinaryRuntime(std::string path)
    {
        std::ofstream ofs(path, std::ios::binary | std::ios::out);
        mo::BinarySaver bar(ofs);
        mo::ISaveVisitor& visitor = bar;
        auto data = TestData<std::shared_ptr<T>>::init();
        visitor(&data, "value0");
    }

    void loadBinaryCereal(std::string path)
    {
        std::ifstream ifs(path, std::ios::binary | std::ios::in);
        cereal::BinaryInputArchive bar(ifs);
        std::shared_ptr<T> tmp;
        std::string error;
        try
        {
            bar(tmp);
        }
        catch (const std::exception& e)
        {
            error = e.what();
        }

        ASSERT_NE(tmp, nullptr) << "Unable to load data";
        auto data = TestData<std::shared_ptr<T>>::init();
        ifs.seekg(0);
        EXPECT_PRED_FORMAT2(TestData<std::shared_ptr<T>>::Compare, *data, *tmp)
            << "Failed to use cereal to deserialize a binary blob. Got:\n"
            << ifs.rdbuf() << "\nExpected:\n"
            << binaryCerealStr();
    }

    void loadBinaryRuntime(std::string path)
    {
        std::ifstream ifs(path, std::ios::binary | std::ios::in);
        mo::BinaryLoader bar(ifs);
        mo::ILoadVisitor& visitor = bar;

        std::shared_ptr<T> tmp;
        visitor(&tmp, "value0");
        ASSERT_NE(tmp, nullptr) << "Unable to load data";
        auto data = TestData<std::shared_ptr<T>>::init();
        ifs.seekg(0);
        EXPECT_PRED_FORMAT2(TestData<std::shared_ptr<T>>::Compare, *data, *tmp)
            << "Failed to use runtime reflection to deserialize a binary blob.\n"
            << ifs.rdbuf();
    }

    void testBinary()
    {
        std::shared_ptr<T> data = TestData<std::shared_ptr<T>>::init();
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

    void testJsonCerealSaveCerealLoad()
    {
        std::shared_ptr<T> data = TestData<std::shared_ptr<T>>::init();
        std::stringstream stat;
        {
            cereal::JSONOutputArchive ar(stat);
            ar(data);
        }

        {
            cereal::JSONInputArchive ar(stat);
            std::shared_ptr<T> loaded;
            ar(loaded);
            EXPECT_NE(loaded, nullptr) << "Failed to deserialize data \n" << stat.str();
            EXPECT_NE(loaded, data);
        }
    }

    void testJsonCerealSaveRuntimeReflectionLoad()
    {
        std::shared_ptr<T> data = TestData<std::shared_ptr<T>>::init();
        std::stringstream stat;
        {
            cereal::JSONOutputArchive ar(stat);
            ar(data);
        }

        {
            std::stringstream ss1(stat.str());
            mo::JSONLoader reader(ss1);
            mo::ILoadVisitor& visitor = reader;
            std::shared_ptr<T> loaded;
            std::string error;
            try
            {
                visitor(&loaded);
            }
            catch (const std::exception& e)
            {
                error = e.what();
            }
            EXPECT_NE(loaded, nullptr) << "Unable to use runtime reflection to load from a json created with cereal "
                                       << error << "\n"
                                       << stat.str();
        }
    }

    void testJsonRuntimeReflectionSaveCerealLoad()
    {
        std::shared_ptr<T> data = TestData<std::shared_ptr<T>>::init();
        std::stringstream sstream;
        {
            mo::JSONSaver saver_(sstream);
            mo::ISaveVisitor& saver = saver_;
            saver(&data);
        }
        {
            bool success = true;
            std::string error;
            try
            {
                cereal::JSONInputArchive ar(sstream);
                std::shared_ptr<T> loaded;
                ar(loaded);
                EXPECT_NE(loaded, nullptr);
                EXPECT_EQ(*loaded, *data) << "Failed to correctly deserialize \n" << sstream.str();
            }
            catch (cereal::RapidJSONException e)
            {
                error = e.what();
                success = false;
            }
            catch (const std::exception& e)
            {
                error = e.what();
                success = false;
            }
            catch (...)
            {
                success = false;
            }
            if (!success)
            {
                std::stringstream savestream;
                {
                    cereal::JSONOutputArchive ar(savestream);
                    ar(data);
                }

                EXPECT_TRUE(success) << "Execption thrown when trying to deserialize. " << error << "\n"
                                     << sstream.str() << "\ncereal expects\n"
                                     << savestream.str();
            }
        }
    }

    void testJsonRuntimeReflectionSaveRuntimeLoad()
    {
        std::shared_ptr<T> data = TestData<std::shared_ptr<T>>::init();
        std::stringstream sstream;
        {
            mo::JSONSaver saver_(sstream);
            mo::ISaveVisitor& saver = saver_;
            saver(&data);
        }
        {

            std::stringstream ss1(sstream.str());
            mo::JSONLoader reader(ss1);
            mo::ILoadVisitor& visitor = reader;
            std::shared_ptr<T> loaded;
            visitor(&loaded);
            EXPECT_EQ(*loaded, *data) << "Unable to use runtime reflection to load from a json created with cereal\n"
                                      << sstream.str();
        }
    }
};

template <class T>
struct SerializationBinary : Serialization<T>
{
};

template <class T>
struct SerializationJson : Serialization<T>
{
};

TYPED_TEST_SUITE_P(SerializationBinary);
TYPED_TEST_SUITE_P(SerializationJson);

TYPED_TEST_P(SerializationBinary, CerealSave)
{
    this->saveBinaryCereal("cereal_test.bin");
}

TYPED_TEST_P(SerializationBinary, RuntimeSave)
{
    this->saveBinaryRuntime("runtime_test.bin");
}

TYPED_TEST_P(SerializationBinary, CerealSaveCerealLoad)
{
    this->loadBinaryCereal("cereal_test.bin");
}

TYPED_TEST_P(SerializationBinary, RuntimeSaveCerealLoad)
{
    this->loadBinaryCereal("runtime_test.bin");
}

TYPED_TEST_P(SerializationBinary, RuntimeSaveRuntimeLoad)
{
    this->loadBinaryRuntime("runtime_test.bin");
}

TYPED_TEST_P(SerializationBinary, CerealSaveRuntimeLoad)
{
    this->loadBinaryRuntime("cereal_test.bin");
}

REGISTER_TYPED_TEST_SUITE_P(SerializationBinary,
                            CerealSave,
                            RuntimeSave,
                            CerealSaveCerealLoad,
                            RuntimeSaveRuntimeLoad,
                            RuntimeSaveCerealLoad,
                            CerealSaveRuntimeLoad);

INSTANTIATE_TYPED_TEST_SUITE_P(serialization_binary, SerializationBinary, RuntimeReflectionTypeTest);

TYPED_TEST_P(SerializationJson, CerealSaveCerealLoad)
{
    this->testJsonCerealSaveCerealLoad();
}

TYPED_TEST_P(SerializationJson, CerealSaveRuntimeLoad)
{
    this->testJsonCerealSaveRuntimeReflectionLoad();
}

TYPED_TEST_P(SerializationJson, RuntimeSaveCerealLoad)
{
    this->testJsonRuntimeReflectionSaveCerealLoad();
}

TYPED_TEST_P(SerializationJson, RuntimeSaveRuntimeLoad)
{
    this->testJsonRuntimeReflectionSaveRuntimeLoad();
}

REGISTER_TYPED_TEST_SUITE_P(
    SerializationJson, CerealSaveCerealLoad, CerealSaveRuntimeLoad, RuntimeSaveCerealLoad, RuntimeSaveRuntimeLoad);

INSTANTIATE_TYPED_TEST_SUITE_P(serialization_json, SerializationJson, RuntimeReflectionTypeTest);

struct RawPointerTestStruct
{
    std::vector<std::shared_ptr<Vec>> owning;
    std::vector<Vec*> raw;
};

namespace ct
{
    REFLECT_BEGIN(RawPointerTestStruct)
        PUBLIC_ACCESS(owning)
        PUBLIC_ACCESS(raw)
    REFLECT_END;
} // namespace ct

template <>
struct TestData<RawPointerTestStruct>
{
    static RawPointerTestStruct init()
    {
        RawPointerTestStruct out;
        out.owning = TestData<std::vector<std::shared_ptr<Vec>>>::init();
        for (size_t i = 0; i < out.owning.size(); ++i)
        {
            out.raw.push_back(out.owning[i].get());
        }
        return out;
    }

    static testing::AssertionResult Compare(const char* lhs_expression,
                                            const char* rhs_expression,
                                            const RawPointerTestStruct& lhs,
                                            const RawPointerTestStruct& rhs)
    {
        if (lhs.owning.size() != rhs.owning.size())
        {
            return testing::internal::EqFailure(
                "lhs.owning.size()",
                "rhs.owning.size()",
                testing::internal::FormatForComparisonFailureMessage(lhs.owning.size(), rhs.owning.size()),
                testing::internal::FormatForComparisonFailureMessage(rhs.owning.size(), lhs.owning.size()),
                false);
        }
        if (lhs.owning.size() != 4)
        {
            return testing::internal::EqFailure(
                "lhs.owning.size()",
                "4",
                testing::internal::FormatForComparisonFailureMessage(lhs.owning.size(), 4),
                testing::internal::FormatForComparisonFailureMessage(4, lhs.owning.size()),
                false);
        }
        for (size_t i = 0; i < 4; ++i)
        {
            if (*(lhs.owning[i]) != *(rhs.owning[i]))
            {
                return testing::internal::EqFailure(
                    "lhs.owning[i]",
                    "rhs.owning[i]",
                    testing::internal::FormatForComparisonFailureMessage(*(lhs.owning[i]), *(rhs.owning[i])),
                    testing::internal::FormatForComparisonFailureMessage(*(rhs.owning[i]), *(lhs.owning[i])),
                    false);
            }
        }
        if (rhs.owning[0] != rhs.owning[3])
        {
            return testing::internal::EqFailure(
                "rhs.owning[0]",
                "rhs.owning[3]",
                testing::internal::FormatForComparisonFailureMessage(rhs.owning[0], rhs.owning[3]),
                testing::internal::FormatForComparisonFailureMessage(rhs.owning[3], rhs.owning[0]),
                false);
        }
        for (size_t i = 0; i < rhs.owning.size(); ++i)
        {
            if (rhs.owning[i].get() != rhs.raw[i])
            {
                return testing::internal::EqFailure(
                    "rhs.owning[i].get()",
                    "rhs.raw[i]",
                    testing::internal::FormatForComparisonFailureMessage(rhs.owning[i].get(), rhs.raw[i]),
                    testing::internal::FormatForComparisonFailureMessage(rhs.raw[i], rhs.owning[i].get()),
                    false);
            }
            if (lhs.owning[i].get() != lhs.raw[i])
            {
                return testing::internal::EqFailure(
                    "lhs.owning[i].get()",
                    "lhs.raw[i]",
                    testing::internal::FormatForComparisonFailureMessage(lhs.owning[i].get(), lhs.raw[i]),
                    testing::internal::FormatForComparisonFailureMessage(lhs.raw[i], lhs.owning[i].get()),
                    false);
            }
        }

        return testing::AssertionSuccess();
    }
};

/*TEST(serialization, rawPointerUpdate)
{
    Serialization<RawPointerTestStruct>::saveBinaryRuntime("test.bin");
    Serialization<RawPointerTestStruct>::loadBinaryRuntime("test.bin");
}*/

namespace
{
    const size_t kRawArraySize = 5;

    template <class T>
    void expectArrayEquals(const T* expected, const ct::TArrayView<T>& loaded)
    {
        for (size_t i = 0; i < kRawArraySize; ++i)
        {
            EXPECT_EQ(expected[i], loaded[i]) << "element " << i;
        }
    }
} // namespace

// The removed mo::ArrayContainerTrait adapted a raw (pointer, size) span —
// externally owned memory with no owning object — into a load/save container.
// The live code covering that use case is ct::TArrayView<T> (TArrayView<T, -1>
// holds the data pointer and the size) together with
// TTraits<ct::TArrayView<T>, 4, void> in visitor_traits/array_adapter.hpp:
// load/save move exactly view.size() elements to/from the memory the view
// points at. This is a different use case than std::array (an owning
// fixed-size container, covered by stdArrayBinaryRoundtrip / stdArrayJson
// below and the testTypes entry in common.hpp — see the comments there for
// why std::array stays out of the typed cross-format suites).
TEST(serialization, rawArrayViewBinary)
{
    static float src[kRawArraySize] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    static float dst[kRawArraySize] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    ct::TArrayView<float> view(src, kRawArraySize);
    ct::TArrayView<float> loaded(dst, kRawArraySize);

    std::stringstream stream;
    {
        mo::BinarySaver saver(stream);
        mo::ISaveVisitor& visitor = saver;
        visitor(&view, "data");
    }
    {
        mo::BinaryLoader loader(stream);
        mo::ILoadVisitor& visitor = loader;
        visitor(&loaded, "data");
    }
    expectArrayEquals(src, loaded);
}

// The removed trait also took a const pointer for save-only use; the
// ct::TArrayView<const T> specialization keeps that asymmetry: save works,
// load throws.
TEST(serialization, rawArrayViewConstSave)
{
    static float src[kRawArraySize] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    static float dst[kRawArraySize] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    const ct::TArrayView<const float> view(src, kRawArraySize);
    ct::TArrayView<float> loaded(dst, kRawArraySize);

    std::stringstream stream;
    {
        mo::BinarySaver saver(stream);
        mo::ISaveVisitor& visitor = saver;
        visitor(&view, "data");
    }
    {
        mo::BinaryLoader loader(stream);
        mo::ILoadVisitor& visitor = loader;
        visitor(&loaded, "data");
    }
    expectArrayEquals(src, loaded);
}

// Mo-own roundtrip for std::array (TTraits<std::array<T, N>, 4, void>).
// Deliberately not part of the typed cross-format suites: mo's container
// binary path frames containers with a uint64 count (BinarySaver/Loader
// IContainerTraits ops), while cereal writes std::array as raw elements with
// no count, so the two formats are not byte-compatible.
TEST(serialization, stdArrayBinaryRoundtrip)
{
    std::array<float, 4> data{1.0F, 2.0F, 3.0F, 4.0F};
    std::array<float, 4> loaded{0.0F, 0.0F, 0.0F, 0.0F};

    std::stringstream stream;
    {
        mo::BinarySaver saver(stream);
        mo::ISaveVisitor& visitor = saver;
        visitor(&data, "data");
    }
    {
        mo::BinaryLoader loader(stream);
        mo::ILoadVisitor& visitor = loader;
        visitor(&loaded, "data");
    }
    for (size_t i = 0; i < loaded.size(); ++i)
    {
        EXPECT_EQ(data[i], loaded[i]) << "element " << i;
    }
}

// std::array in JSON. After the doubled array framing was removed from
// JSONSaver::writePod / JSONLoader::readPod, mo emits a single array
// ({"value0": [1, 2, 3, 4]} at top level) — the same layout cereal uses for
// std::vector, which is why the typed vector JSON cross-tests pass.
//
// mo-own roundtrip only, deliberately: cereal has NO array layout for
// std::array in JSON. Its array.hpp save is `for (i) ar(i)` with no SizeTag,
// so a top-level std::array<float,4> comes out as
// {"value0": {"value0": 1, "value1": 2, "value2": 3, "value3": 4}} — a flat
// nested object of scalars. mo's loader reads that as a 4-member container
// whose first member is not an array (throws), and cereal reads mo's
// [1,2,3,4] as a single number (throws). Same incompatibility class as the
// binary path (cereal array = raw elements, mo container = uint64 count
// prefix), see stdArrayBinaryRoundtrip above.
TEST(serialization, stdArrayJson)
{
    std::array<float, 4> data{1.0F, 2.0F, 3.0F, 4.0F};

    std::stringstream ss;
    {
        mo::JSONSaver saver(ss);
        mo::ISaveVisitor& visitor = saver;
        visitor(&data);
    }
    std::array<float, 4> loaded{};
    std::stringstream ss1(ss.str());
    mo::JSONLoader reader(ss1);
    mo::ILoadVisitor& visitor = reader;
    visitor(&loaded);
    for (size_t i = 0; i < data.size(); ++i)
    {
        EXPECT_EQ(data[i], loaded[i]) << "element " << i << ":\n"
                                      << ss.str();
    }
}

// mo-only JSON roundtrip for the raw-span view (ct::TArrayView has no cereal
// equivalent). Exercises the view trait's load/save in the single-array
// layout; the binary counterpart is rawArrayViewBinary above.
TEST(serialization, rawArrayViewJson)
{
    static float src[kRawArraySize] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    static float dst[kRawArraySize] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    ct::TArrayView<float> view(src, kRawArraySize);
    ct::TArrayView<float> loaded(dst, kRawArraySize);

    std::stringstream ss;
    {
        mo::JSONSaver saver(ss);
        mo::ISaveVisitor& visitor = saver;
        visitor(&view);
    }
    std::stringstream ss1(ss.str());
    mo::JSONLoader reader(ss1);
    mo::ILoadVisitor& visitor = reader;
    visitor(&loaded);
    expectArrayEquals(src, loaded);
}

template <class T>
void testBinarySpeed(size_t count)
{
    std::vector<T> vec(count);
    std::cout << "------ " << mo::TypeInfo::create<T>().name() << " ------ " << std::endl;
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

TEST(serialization, binary_performance)
{
    int size = 9;
    bool success = false;
    while (!success)
    {
        try
        {
            testBinarySpeed<float>(static_cast<size_t>(std::exp(size)));
            success = true;
        }
        catch (std::bad_alloc)
        {
            size -= 1;
        }
    }
    success = false;
    size = 7;
    while (!success)
    {
        try
        {
            testBinarySpeed<TestPodStruct>(static_cast<size_t>(std::exp(size)));
            success = true;
        }
        catch (std::bad_alloc)
        {
            size -= 1;
        }
    }
}
