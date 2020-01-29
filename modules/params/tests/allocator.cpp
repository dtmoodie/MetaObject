#include "../../runtime_reflection/tests/common.hpp"

#include <MetaObject/core/detail/allocator_policies/Pool.hpp>
#include <MetaObject/core/detail/allocator_policies/Stack.hpp>
#include <MetaObject/params/TDataContainer.hpp>
#include <MetaObject/params/TParam.hpp>

#include <ct/types/TArrayView.hpp>

#include <cereal/types/vector.hpp>

#include <gtest/gtest.h>

using namespace mo;

struct SaveStream
{
    SaveStream(ct::TArrayView<uint8_t> buf);

    template <class T>
    auto write(const T* data, size_t num_elems = 1) ->
        typename std::enable_if<std::is_trivially_copyable<T>::value, bool>::type
    {
        const auto bytes = sizeof(T) * num_elems;
        MO_ASSERT(bytes <= m_serialization_buffer.size());
        const bool serialize_in_place = ptrCast<uint8_t>(data) == m_serialization_buffer.data();
        if (!serialize_in_place)
        {
            memcpy(m_serialization_buffer.data(), data, bytes);
        }
        auto next = m_serialization_buffer.slice(bytes);
        MO_ASSERT_EQ(next.size(), m_serialization_buffer.size() - bytes);
        MO_ASSERT_EQ(next.data(), m_serialization_buffer.data() + bytes);
        m_serialization_buffer = next;
        return serialize_in_place;
    }

    template <class T>
    auto write(const T* data, size_t num_elems = 1) ->
        typename std::enable_if<!std::is_trivially_copyable<T>::value, bool>::type
    {
        for (size_t i = 0; i < num_elems; ++i)
        {
            save(*this, data[i]);
        }
        return false;
    }

    ct::TArrayView<uint8_t> buffer()
    {
        return m_serialization_buffer;
    }

  private:
    ct::TArrayView<uint8_t> m_serialization_buffer;
};

struct LoadStream
{
    LoadStream(ct::TArrayView<const uint8_t> buf);

    template <class T>
    auto read(T* data, size_t num_elems = 1) ->
        typename std::enable_if<std::is_trivially_copyable<T>::value, bool>::type
    {
        const auto bytes = sizeof(T) * num_elems;
        MO_ASSERT(bytes <= m_serialization_buffer.size());
        const bool serialize_in_place = ptrCast<uint8_t>(data) == m_serialization_buffer.data();
        if (!serialize_in_place)
        {
            memcpy(data, m_serialization_buffer.data(), bytes);
        }
        auto next = m_serialization_buffer.slice(bytes);
        MO_ASSERT_EQ(next.size(), m_serialization_buffer.size() - bytes);
        MO_ASSERT_EQ(next.data(), m_serialization_buffer.data() + bytes);
        m_serialization_buffer = next;
        return serialize_in_place;
    }

    template <class T>
    auto read(T* data, size_t num_elems = 1) ->
        typename std::enable_if<!std::is_trivially_copyable<T>::value, bool>::type
    {
        for (size_t i = 0; i < num_elems; ++i)
        {
            load(*this, data[i]);
        }
        return false;
    }

    const void* next(const size_t bytes)
    {
        MO_ASSERT(bytes <= m_serialization_buffer.size());
        const void* ret = m_serialization_buffer.data();
        m_serialization_buffer = m_serialization_buffer.slice(bytes);
        return ret;
    }

  private:
    ct::TArrayView<const uint8_t> m_serialization_buffer;
};

SaveStream::SaveStream(ct::TArrayView<uint8_t> buf)
    : m_serialization_buffer(buf)
{
}

LoadStream::LoadStream(ct::TArrayView<const uint8_t> buf)
    : m_serialization_buffer(buf)
{
}

template <class T, class A>
uint32_t serializedSize(const std::vector<T, A>& data)
{
    return sizeof(T) * data.size();
}

template <class T>
uint32_t serializedSize(const boost::optional<T>& data)
{
    if (data)
    {
        return sizeof(bool) + sizeof(T);
    }
    return sizeof(bool);
}

template <class T>
uint32_t serializedSize(const T&)
{
    static_assert(std::is_trivially_copyable<T>::value, "Must be trivially copyable");
    return sizeof(T);
}

uint32_t serializedSize(const mo::Header& data)
{
    return serializedSize(data.frame_number) + serializedSize(data.timestamp);
}

template <class T>
void save(SaveStream& stream, const boost::optional<T>& data)
{
    bool status = data.is_initialized();
    stream.write(&status);
    if (status)
    {
        stream.write(&*data);
    }
}

template <class T>
auto saveHelper(SaveStream& stream, const T& data, ct::Indexer<0> idx) -> ct::EnableIfReflected<T>
{
    auto ptr = ct::Reflect<T>::getPtr(idx);
    stream.write(&ptr.get(data), 1);
}

template <class T, ct::index_t I>
auto saveHelper(SaveStream& stream, const T& data, ct::Indexer<I> idx) -> ct::EnableIfReflected<T>
{
    auto ptr = ct::Reflect<T>::getPtr(idx);
    stream.write(&ptr.get(data), 1);
    saveHelper(stream, data, --idx);
}

template <class T>
auto save(SaveStream& stream, const T& data) -> ct::EnableIfReflected<T>
{
    saveHelper(stream, data, ct::Reflect<T>::end());
}

template <class T>
auto loadHelper(LoadStream& stream, T& data, ct::Indexer<0> idx) -> ct::EnableIfReflected<T>
{
    auto ptr = ct::Reflect<T>::getPtr(idx);
    stream.read(&ptr.set(data), 1);
}

template <class T, ct::index_t I>
auto loadHelper(LoadStream& stream, T& data, ct::Indexer<I> idx) -> ct::EnableIfReflected<T>
{
    auto ptr = ct::Reflect<T>::getPtr(idx);
    stream.read(&ptr.set(data), 1);
    loadHelper(stream, data, --idx);
}

template <class T>
auto load(LoadStream& stream, T& data) -> ct::EnableIfReflected<T>
{
    loadHelper(stream, data, ct::Reflect<T>::end());
}

template <class T>
void load(LoadStream& stream, boost::optional<T>& data)
{
    bool status = false;
    stream.read(&status);
    if (status)
    {
        data = T();
        stream.read(&*data);
    }
}

template <class T, class A>
std::shared_ptr<ParamAllocator::SerializationBuffer> save(const TDataContainer<std::vector<T, A>>& container,
                                                          bool* in_place = nullptr)
{
    const auto header_size = serializedSize(container.header) + sizeof(uint32_t);
    auto allocator = container.getAllocator();
    auto data = container.data.data();
    auto binary_serialization_buffer = allocator->allocateSerialization(header_size, 0, data);
    SaveStream stream(binary_serialization_buffer->slice(0));

    stream.write(&container.header);
    const uint32_t sz = container.data.size();
    stream.write(&sz);
    if (in_place)
    {
        *in_place = stream.write(data, container.data.size());
    }
    else
    {
        stream.write(data, container.data.size());
    }
    return binary_serialization_buffer;
}

template <class T, class A>
void load(TDataContainer<std::vector<T, A>>& container, const std::shared_ptr<ParamAllocator::SerializationBuffer>& buf)
{
    LoadStream stream(buf->slice(0));
    stream.read(&container.header);
    uint32_t sz = 0;
    stream.read(&sz);
    container.data.resize(sz);
    stream.read(container.data.data(), sz);
}

template <class T>
void load(TDataContainer<ct::TArrayView<const T>>& container,
          const std::shared_ptr<ParamAllocator::SerializationBuffer>& buf)
{
    LoadStream stream(buf->slice(0));
    stream.read(&container.header);
    uint32_t sz = 0;
    stream.read(&sz);
    auto ptr = stream.next(sz * sizeof(T));
    container.data = ct::TArrayView<const T>(ct::ptrCast<T>(ptr), sz);
    container.owning = buf;
}

bool serializeWithHeader(const void* header,
                         size_t header_size,
                         const std::vector<float, TVectorAllocator<float>>& data)
{
    auto allocator = data.get_allocator().getAllocator();
    auto serialization_buffer = allocator->allocateSerialization(header_size, 0, data.data());

    EXPECT_NE(serialization_buffer->data(), nullptr);
    SaveStream stream(*serialization_buffer);
    stream.write(ptrCast<uint8_t>(header), header_size);
    auto src = data.data();
    if (ptrCast<>(src) != stream.buffer().data())
    {
        stream.write(data.data(), data.size());
        return false;
    }
    return true;
}

TEST(serialization_aware_allocator, no_header)
{
    auto alloc = ParamAllocator::create();
    TDataContainer<std::vector<float>> container(alloc);
    container.data.reserve(static_cast<size_t>(1e4));
    ASSERT_EQ(container.data.capacity(), 1e4);
    for (int i = 0; i < 1e4; ++i)
    {
        container.data.push_back(static_cast<float>(i));
    }

    auto allocator = container.data.get_allocator().getAllocator();
    auto serialization_buffer = allocator->allocateSerialization(0, 0, container.data.data());

    ASSERT_NE(serialization_buffer->data(), nullptr);
    SaveStream stream(*serialization_buffer);
    auto src = container.data.data();
    ASSERT_EQ(ptrCast<>(src), stream.buffer().data());
}

TEST(serialization_aware_allocator, predefined_pad)
{
    auto alloc = ParamAllocator::create();
    TDataContainer<std::vector<float>> container(alloc);
    container.getAllocator()->setPadding(sizeof(uint32_t));
    container.data.reserve(static_cast<size_t>(1e4));
    ASSERT_EQ(container.data.capacity(), 1e4);
    for (int i = 0; i < 1e4; ++i)
    {
        container.data.push_back(static_cast<float>(i));
    }

    auto allocator = container.data.get_allocator().getAllocator();
    auto serialization_buffer = allocator->allocateSerialization(sizeof(uint32_t), 0, container.data.data());

    ASSERT_NE(serialization_buffer->data(), nullptr);
    SaveStream stream(*serialization_buffer);
    auto src = container.data.data();
    uint32_t sz = static_cast<uint32_t>(container.data.size());
    stream.write(&sz);
    ASSERT_EQ(ptrCast<void>(ptrCast<>(src)), ptrCast<void>(stream.buffer().data()));
    ASSERT_TRUE(stream.write(container.data.data(), sz));
}

TEST(serialization_aware_allocator, container)
{
    auto alloc = ParamAllocator::create();
    TDataContainer<std::vector<float>> container(alloc);
    container.data.reserve(static_cast<size_t>(1e4));
    ASSERT_EQ(container.data.capacity(), 1e4);
    for (int i = 0; i < 1e4; ++i)
    {
        container.data.push_back(static_cast<float>(i));
    }
    float header;
    ASSERT_FALSE(serializeWithHeader(&header, sizeof(float), container.data));
}

void serializeContainerOverpad(Allocator::Ptr_t allocator_)
{
    auto allocator = ParamAllocator::create(allocator_);
    TDataContainer<std::vector<float>> container(allocator);
    container.getAllocator()->setPadding(10);
    container.data.reserve(static_cast<size_t>(1e4));
    ASSERT_EQ(container.data.capacity(), 1e4);
    for (int i = 0; i < 1e4; ++i)
    {
        container.data.push_back(static_cast<float>(i));
    }
    float header;
    ASSERT_TRUE(serializeWithHeader(&header, sizeof(float), container.data));
}

TEST(serialization_aware_allocator, overpad_default_allocator)
{
    serializeContainerOverpad(Allocator::getDefault());
}

TEST(serialization_aware_allocator, overpad_pool_allocator)
{
    auto alloc = std::make_shared<mo::PoolPolicy<mo::CPU>>();
    serializeContainerOverpad(alloc);
}

TEST(serialization_aware_allocator, overpad_stack_allocator)
{
    auto alloc = std::make_shared<mo::PoolPolicy<mo::CPU>>();
    serializeContainerOverpad(alloc);
}

void serializeContainerCreatedFromParam(Allocator::Ptr_t allocator_)
{
    auto allocator = ParamAllocator::create(allocator_);
    TDataContainer<std::vector<float>> container0(allocator);

    float header;
    container0.data.reserve(static_cast<size_t>(1e4));
    for (int i = 0; i < 1e4; ++i)
    {
        container0.data.push_back(static_cast<float>(i));
    }
    ASSERT_FALSE(serializeWithHeader(&header, sizeof(float), container0.data));
    ASSERT_EQ(container0.data.capacity(), 1e4);
    for (int i = 0; i < 1e4; ++i)
    {
        ASSERT_EQ(container0.data[i], i);
    }

    TDataContainer<std::vector<float>> container1(allocator);
    container1.data.reserve(static_cast<size_t>(1e4));
    for (int i = 0; i < 1e4; ++i)
    {
        container1.data.push_back(static_cast<float>(i));
    }
    ASSERT_EQ(container1.data.capacity(), 1e4);
    ASSERT_TRUE(serializeWithHeader(&header, sizeof(float), container1.data));
    for (int i = 0; i < 1e4; ++i)
    {
        ASSERT_EQ(container1.data[i], i);
    }
}

TEST(serialization_aware_allocator, container_created_from_param_default_allocator)
{
    serializeContainerCreatedFromParam(Allocator::getDefault());
}

TEST(serialization_aware_allocator, container_created_from_param_pool_allocator)
{
    auto alloc = std::make_shared<mo::PoolPolicy<mo::CPU>>();
    serializeContainerCreatedFromParam(alloc);
}

TEST(serialization_aware_allocator, container_created_from_param_stack_allocator)
{
    auto alloc = std::make_shared<mo::StackPolicy<mo::CPU>>();
    serializeContainerCreatedFromParam(alloc);
}

void serializeAndDeserializeFromParam(Allocator::Ptr_t allocator)
{
    TParam<std::vector<float>> param;
    param.setAllocator(allocator);

    for (size_t sz = 100; sz <= 10000; sz += 100)
    {
        auto data = param.create();
        data->header.timestamp = mo::Time(10 * mo::ms);
        data->header.frame_number = 15;
        data->data.reserve(sz);
        ASSERT_EQ(data->data.capacity(), sz);
        for (int i = 0; i < sz; ++i)
        {
            data->data.push_back(static_cast<float>(i));
        }
        bool in_place = false;
        auto buf = save(*data, &in_place);
        if (sz == 100)
        {
            ASSERT_FALSE(in_place);
        }
        else
        {
            ASSERT_TRUE(in_place);
        }

        for (int i = 0; i < data->data.size(); ++i)
        {
            ASSERT_EQ(data->data[i], i);
        }

        auto alloc = ParamAllocator::create(allocator);
        TDataContainer<std::vector<float>> load_container(alloc);
        load(load_container, buf);
        ASSERT_TRUE(load_container.header.timestamp);
        ASSERT_EQ(load_container.header.frame_number, 15);
        ASSERT_EQ(*load_container.header.timestamp, mo::Time(mo::ms * 10));
        ASSERT_EQ(load_container.data.size(), data->data.size());

        // Verify that data is unchanged
        for (int i = 0; i < data->data.size(); ++i)
        {
            ASSERT_EQ(data->data[i], i);
        }

        for (int i = 0; i < load_container.data.size(); ++i)
        {
            ASSERT_EQ(load_container.data[i], i);
        }
    }
}

TEST(serialization_aware_allocator, container_from_param_default_allocator)
{
    serializeAndDeserializeFromParam(mo::Allocator::getDefault());
}

TEST(serialization_aware_allocator, container_from_param_pool_allocator)
{
    auto alloc = std::make_shared<mo::PoolPolicy<mo::CPU>>();
    serializeAndDeserializeFromParam(alloc);
}

TEST(serialization_aware_allocator, container_from_param_stack_allocator)
{
    auto alloc = std::make_shared<mo::StackPolicy<mo::CPU>>();
    serializeAndDeserializeFromParam(alloc);
}

TEST(serialization_aware_allocator, load_wrap)
{
    auto allocator = mo::Allocator::getDefault();
    TParam<std::vector<float>> param;
    param.setAllocator(allocator);

    for (size_t sz = 100; sz <= 10000; sz += 100)
    {
        auto data = param.create();
        data->header.timestamp = mo::Time(10 * mo::ms);
        data->header.frame_number = 15;
        data->data.reserve(sz);
        ASSERT_EQ(data->data.capacity(), sz);
        for (int i = 0; i < sz; ++i)
        {
            data->data.push_back(static_cast<float>(i));
        }
        bool in_place = false;
        auto buf = save(*data, &in_place);
        if (sz == 100)
        {
            ASSERT_FALSE(in_place);
        }
        else
        {
            ASSERT_TRUE(in_place);
        }

        for (int i = 0; i < data->data.size(); ++i)
        {
            ASSERT_EQ(data->data[i], i);
        }

        auto alloc = ParamAllocator::create(allocator);
        TDataContainer<ct::TArrayView<const float>> load_container(alloc);
        load(load_container, buf);
        ASSERT_TRUE(load_container.header.timestamp);
        ASSERT_EQ(load_container.header.frame_number, 15);
        ASSERT_EQ(*load_container.header.timestamp, mo::Time(mo::ms * 10));
        ASSERT_EQ(load_container.data.size(), data->data.size());

        // Verify that data is unchanged
        for (int i = 0; i < data->data.size(); ++i)
        {
            ASSERT_EQ(data->data[i], i);
        }

        for (int i = 0; i < load_container.data.size(); ++i)
        {
            ASSERT_EQ(load_container.data[i], i);
        }

        ASSERT_GT(ct::ptrCast<>(load_container.data.begin()), buf->data()) << "Ensure view lower bound in data buffer";
        ASSERT_LE(ct::ptrCast<>(load_container.data.end()), (buf->data() + buf->size()))
            << "Ensure view upper bound in data buffer";

        ASSERT_TRUE(reinterpret_cast<size_t>(load_container.data.data()) % sizeof(float) == 0)
            << "Ensuring alignment of elements in view is correct";
    }
}

TEST(serialization_aware_allocator, vector_of_reflected)
{
    TParam<std::vector<TestPodStruct>> param;
    {
        auto container = param.create(100);
        container->data.resize(100);
        for (size_t i = 0; i < container->data.size(); ++i)
        {
            container->data[i].x = static_cast<float>(1.0 * i);
            container->data[i].y = static_cast<float>(2.0 * i);
            container->data[i].z = static_cast<float>(3.0 * i);
            container->data[i].id = static_cast<uint32_t>(i);
        }
        bool in_place = false;
        auto buf = save(*container, &in_place);
        ASSERT_FALSE(in_place);
        for (size_t i = 0; i < container->data.size(); ++i)
        {
            ASSERT_EQ(container->data[i].x, 1.0 * i);
            ASSERT_EQ(container->data[i].y, 2.0 * i);
            ASSERT_EQ(container->data[i].z, 3.0 * i);
            ASSERT_EQ(container->data[i].id, i);
        }
    }
    {
        auto container = param.create(100);
        container->data.resize(100);
        for (size_t i = 0; i < container->data.size(); ++i)
        {
            container->data[i].x = static_cast<float>(1.0 * i);
            container->data[i].y = static_cast<float>(2.0 * i);
            container->data[i].z = static_cast<float>(3.0 * i);
            container->data[i].id = static_cast<uint32_t>(i);
        }
        bool in_place = false;
        auto buf = save(*container, &in_place);
        ASSERT_TRUE(in_place);

        TDataContainer<std::vector<TestPodStruct>> load_container;
        load(load_container, buf);
        for (size_t i = 0; i < load_container.data.size(); ++i)
        {
            ASSERT_EQ(load_container.data[i].x, 1.0 * i);
            ASSERT_EQ(load_container.data[i].y, 2.0 * i);
            ASSERT_EQ(load_container.data[i].z, 3.0 * i);
            ASSERT_EQ(load_container.data[i].id, i);
        }
    }
}

namespace
{
    struct alignas(4 * sizeof(float)) AlignedVector
    {
        REFLECT_INTERNAL_BEGIN(AlignedVector)
            REFLECT_INTERNAL_MEMBER(float, x)
            REFLECT_INTERNAL_MEMBER(float, y)
            REFLECT_INTERNAL_MEMBER(float, z)
        REFLECT_INTERNAL_END;
    };

} // namespace

TEST(serialization_aware_allocator, vector_alignment)
{
    TParam<std::vector<AlignedVector>> param;
    {
        {
            auto container = param.create(100);
            for (size_t i = 0; i < container->data.size(); ++i)
            {
                container->data[i].x = static_cast<float>(1.0 * i);
                container->data[i].y = static_cast<float>(2.0 * i);
                container->data[i].z = static_cast<float>(3.0 * i);
            }

            bool in_place = false;
            auto buf = save(*container, &in_place);
            ASSERT_FALSE(in_place);
            auto data = container->data.data();
            ASSERT_EQ(reinterpret_cast<size_t>(data) % (4 * sizeof(float)), 0);
        }
        {
            auto container = param.create(100);
            for (size_t i = 0; i < container->data.size(); ++i)
            {
                container->data[i].x = static_cast<float>(1.0 * i);
                container->data[i].y = static_cast<float>(2.0 * i);
                container->data[i].z = static_cast<float>(3.0 * i);
            }

            bool in_place = false;
            auto buf = save(*container, &in_place);
            ASSERT_TRUE(in_place);
            auto data = container->data.data();
            ASSERT_EQ(reinterpret_cast<size_t>(data) % (4 * sizeof(float)), 0);
        }
    }
}
