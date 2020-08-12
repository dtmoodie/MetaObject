#include <MetaObject/params/TPublisher.hpp>
#include <MetaObject/params/TSubscriber.hpp>
#include <MetaObject/params/TSubscriberPtr.hpp>

#include <MetaObject/params/buffers/BufferFactory.hpp>
#include <MetaObject/params/buffers/Buffers.hpp>
#include <MetaObject/params/buffers/IBuffer.hpp>
#include <MetaObject/params/buffers/Map.hpp>

#include <boost/fiber/operations.hpp>

#include <boost/thread.hpp>
#include <set>

#include <gtest/gtest.h>

using namespace mo;

namespace
{

    struct buffers : ::testing::Test
    {
        TSubscriber<int> input_param;
        TPublisher<int> param;

        std::shared_ptr<buffer::IBuffer> buffer;
        IAsyncStreamPtr_t m_stream;

        buffers()
        {
            m_stream = IAsyncStream::create();
            param.setStream(*m_stream);
            param.setName("pub");

            input_param.setStream(*m_stream);
            input_param.setName("sub");
        }

        void init(BufferFlags buffer_type)
        {
            ASSERT_EQ(param.getName(), "pub");
            buffer = mo::buffer::IBuffer::create(buffer_type);
            buffer->setInput(&param);
            ASSERT_NE(buffer, nullptr);
            ASSERT_TRUE(input_param.setInput(buffer.get()));
        }

        void testRead()
        {
            int data = 0;
            Header header(mo::ms * 1.0);
            param.publish(1, header);

            ASSERT_EQ(buffer->getSize(), 1);

            ASSERT_NE(input_param.getData(), nullptr);
            ASSERT_TRUE(input_param.getData(data));
            ASSERT_EQ(data, 1);

            ASSERT_NE(input_param.getData(&header), nullptr);

            header = Header(mo::ms * 2.0);
            auto retrieved_data = input_param.getData(&header);
            ASSERT_EQ(retrieved_data, nullptr)
                << "We don't expect to be able to retrieve data when requesting an invalid timestamp " << header
                << " yet we retrieved data at " << retrieved_data->getHeader()
                << " \navailable data: " << buffer->getAvailableHeaders();

            param.publish(2, header);

            ASSERT_EQ(buffer->getSize(), 2);
            ASSERT_TRUE(input_param.getData(data));
            ASSERT_EQ(data, 2);

            header = Header(mo::ms * 5.0);
            param.publish(5, header);
            ASSERT_EQ(buffer->getSize(), 3);

            ASSERT_NE(input_param.getData(), nullptr);
            ASSERT_TRUE(input_param.getData(data));
            ASSERT_EQ(data, 5);

            data = 10;
            header = Header(mo::ms * 5.0);
            ASSERT_TRUE(input_param.getData(data, &header));
            ASSERT_EQ(data, 5);
        }

        void fill(const int32_t num)
        {
            ASSERT_NE(buffer, nullptr);
            buffer->clear();
            ASSERT_EQ(buffer->getSize(), 0);
            int32_t data;

            for (int32_t i = 0; i < num; ++i)
            {
                const auto header = Header(i * ms);
                param.publish(i, header);
                ASSERT_TRUE(input_param.getData(data, &header)) << "Unable to retrive data by header " << header
                                                                << " existing data " << buffer->getAvailableHeaders();
            }
        }

        void testPruning()
        {
            buffer->setFrameBufferCapacity(10);
            fill(100);
            ASSERT_EQ(buffer->getSize(), 10);
        }

        int testMultiThreaded(const bool dropping)
        {
            int read_values = 0;
            buffer->clear();
            bool started = false;
            bool interrupt = false;
            auto read_func = [this, dropping, &read_values, &interrupt, &started]() {
                try
                {
                    int data;
                    std::set<int> read_values_;
                    started = true;
                    for (int i = 0; i < 10000; ++i)
                    {
                        bool result = false;
                        if (!dropping)
                        {
                            for (int j = 0; j < 15 && !result; ++j)
                            {
                                const auto header = Header(i * ms);
                                result = input_param.getData(data, &header);
                                boost::this_fiber::sleep_for(std::chrono::nanoseconds(5));
                            }
                        }
                        else
                        {
                            int new_value;
                            const auto header = Header(i * ms);
                            result = input_param.getData(new_value, &header);
                            if (new_value == data || !result)
                            {
                                continue;
                            }
                            EXPECT_EQ(read_values_.find(new_value), read_values_.end());
                            read_values_.insert(new_value);

                            data = new_value;
                        }

                        EXPECT_EQ((result || dropping), true);
                        if (false == (result || dropping))
                        {
                            return;
                        }

                        if (result)
                        {
                            ASSERT_EQ(data, i);
                            ++read_values;
                        }
                        auto wait = rand() % 5 + 30;
                        boost::this_fiber::sleep_for(std::chrono::nanoseconds(wait));
                        if (interrupt)
                        {
                            return;
                        }
                    }
                    if (dropping)
                    {
                        read_values = static_cast<int>(read_values_.size());
                    }
                }
                catch (...)
                {
                }
            };
            boost::fibers::fiber thread(read_func);
            while (!started)
            {
                boost::this_fiber::sleep_for(std::chrono::nanoseconds(10));
            }
            for (int i = 0; i < 10000; ++i)
            {
                // TODO fix
                param.publish(i, tags::timestamp = i * ms);
                boost::this_fiber::sleep_for(std::chrono::nanoseconds(33));
            }
            thread.join();
            return read_values;
        }
    };
} // namespace

TEST_F(buffers, map_lookup)
{
    using Buffer_t = std::map<Header, IDataContainerConstPtr_t>;
    Buffer_t buffer;
    buffer[mo::Header(1.0 * mo::ms)] = IDataContainerConstPtr_t();
    buffer[mo::Header(2.0 * mo::ms)] = IDataContainerConstPtr_t();
    buffer[mo::Header(3.0 * mo::ms)] = IDataContainerConstPtr_t();

    auto itr = buffer.find(mo::Header(1.0 * mo::ms));
    ASSERT_NE(itr, buffer.end());
    itr = buffer.find(mo::Header(2.0 * mo::ms));
    ASSERT_NE(itr, buffer.end());
    itr = buffer.find(mo::Header(3.0 * mo::ms));
    ASSERT_NE(itr, buffer.end());
}

TEST_F(buffers, map_push)
{
    mo::buffer::Map map;

    mo::TPublisher<int> pub;
    map.setInput(&pub);

    pub.publish(5, mo::Header(1.0 * mo::ms));

    EXPECT_EQ(map.getSize(), 1);
}

TEST_F(buffers, map_retrieve)
{
    mo::buffer::Map map;

    mo::TPublisher<int> pub;
    map.setInput(&pub);
    auto header = mo::Header(1.0 * mo::ms);
    pub.publish(5, header);

    auto data = map.getData(&header);
    EXPECT_TRUE(data);
}

TEST_F(buffers, read_map)
{
    init(BufferFlags::MAP_BUFFER);
    testRead();
    buffer->setFrameBufferCapacity(10);
    fill(100);
    ASSERT_EQ(buffer->getSize(), 100);
    const auto read_values = testMultiThreaded(false);
    ASSERT_EQ(read_values, 10000);
}

TEST_F(buffers, read_stream)
{
    init(BufferFlags::STREAM_BUFFER);
    testRead();
    testPruning();
    const auto read_values = testMultiThreaded(false);
    ASSERT_EQ(read_values, 10000);
}

TEST_F(buffers, blocking_stream)
{
    init(BufferFlags::BLOCKING_STREAM_BUFFER);
    testRead();
    testPruning();
    const auto read_values = testMultiThreaded(false);
    ASSERT_EQ(read_values, 10000);
}

TEST_F(buffers, read_dropping_stream)
{
    init(BufferFlags::DROPPING_STREAM_BUFFER);
    testRead();
    testPruning();
    // TODO more reliable test, fails more than half the time depending on CPU load
    // const auto read_values = testMultiThreaded(true);
    // BOOST_REQUIRE_GT(read_values, 1000);
}

TEST_F(buffers, read_nearest_neighbor)
{
    init(BufferFlags::NEAREST_NEIGHBOR_BUFFER);

    Header header;
    for (int time = 1; time < 10; ++time)
    {
        header = Header(time * mo::ms);
        param.publish(time, header);
    }

    int data = 0;

    header = Header(0.5 * mo::ms);
    EXPECT_TRUE(input_param.getData(data, &header));
    EXPECT_EQ(data, 1) << "Retrieved an incorrect value from the buffer at header " << header
                       << " available data: " << input_param.getAvailableHeaders();

    header = Header(1.4 * mo::ms);
    EXPECT_TRUE(input_param.getData(data, &header));
    EXPECT_EQ(data, 1) << "Retrieved an incorrect value from the buffer at header " << header
                       << " available data: " << input_param.getAvailableHeaders();

    header = Header(1.6 * mo::ms);
    EXPECT_TRUE(input_param.getData(data, &header));
    EXPECT_EQ(data, 2) << "Retrieved an incorrect value from the buffer at header " << header
                       << " available data: " << input_param.getAvailableHeaders();

    header = Header(3.6 * mo::ms);
    EXPECT_TRUE(input_param.getData(data, &header));
    EXPECT_EQ(data, 4) << "Retrieved an incorrect value from the buffer at header " << header
                       << " available data: " << input_param.getAvailableHeaders();

    header = Header(8.4 * mo::ms);
    EXPECT_TRUE(input_param.getData(data, &header));
    EXPECT_EQ(data, 8) << "Retrieved an incorrect value from the buffer at header " << header
                       << " available data: " << input_param.getAvailableHeaders();

    header = Header(8.6 * mo::ms);
    EXPECT_TRUE(input_param.getData(data, &header));
    EXPECT_EQ(data, 9) << "Retrieved an incorrect value from the buffer at header " << header
                       << " available data: " << input_param.getAvailableHeaders();

    testPruning();
}
