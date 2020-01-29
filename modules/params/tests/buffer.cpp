#include <MetaObject/params/ITInputParam.hpp>
#include <MetaObject/params/TInputParam.hpp>
#include <MetaObject/params/TParamPtr.hpp>
#include <MetaObject/params/buffers/BufferFactory.hpp>
#include <MetaObject/params/buffers/Buffers.hpp>
#include <MetaObject/params/buffers/IBuffer.hpp>
#include <MetaObject/thread/fiber_include.hpp>

#include <boost/thread.hpp>
#include <set>

#include <gtest/gtest.h>

using namespace mo;

namespace
{

    struct ParamBufferFixture : ::testing::Test
    {
        int val = 0;
        ITInputParam<int> input_param;
        TParamPtr<int> param;

        std::unique_ptr<buffer::IBuffer> buffer;

        ParamBufferFixture()
            : input_param("sub")
            , param("pub", &val)
        {
            param.setName("pub");
            input_param.setName("sub");
        }

        void init(BufferFlags buffer_type)
        {
            ASSERT_EQ(param.getName(), "pub");
            // ASSERT_EQ(param.getData(), nullptr);

            // buffer.reset(buffer::BufferFactory::createBuffer(&param, buffer_type));
            buffer.reset(mo::buffer::IBuffer::create(buffer_type));
            buffer->setInput(&param);
            ASSERT_NE(buffer, nullptr);
            ASSERT_EQ(input_param.setInput(buffer.get()), true);
        }

        void testRead()
        {
            int data;
            param.updateData(1, Header(Time(mo::ms * 1.0)));

            ASSERT_EQ(buffer->getSize(), 2);
            ASSERT_NE(input_param.getData(Header()), nullptr);
            ASSERT_NE(input_param.getData(Header(Time(mo::ms * 1.0))), nullptr);
            ASSERT_EQ(input_param.getTypedData(&data, Header(Time(mo::ms * 1.0))), true);
            ASSERT_EQ(data, 1);

            param.updateData(2, Header(Time(mo::ms * 2.0)));
            ASSERT_EQ(buffer->getSize(), 3);
            ASSERT_EQ(input_param.getTypedData(&data, Header(Time(mo::ms * 2.0))), true);
            ASSERT_EQ(data, 2);

            param.updateData(5, Header(Time(mo::ms * 5.0)));
            ASSERT_EQ(buffer->getSize(), 4);
            ASSERT_EQ(input_param.getTypedData(&data, Header(Time(mo::ms * 5.0))), true);
            ASSERT_EQ(data, 5);

            ASSERT_NE(input_param.getData(Header()), nullptr);
            ASSERT_EQ(input_param.getTypedData(&data, Header()), true);
            ASSERT_EQ(data, 5);

            data = 10;
            ASSERT_EQ(input_param.getTypedData(&data, Header(mo::ms * 5.0)), true);
            ASSERT_EQ(data, 5);
        }

        void fill(const int32_t num)
        {
            ASSERT_NE(buffer, nullptr);
            buffer->clear();
            int32_t data;
            for (int32_t i = 0; i < num; ++i)
            {
                param.updateData(i, Header(i * ms));
                input_param.getTypedData(&data, Header(i * ms));
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
                                result = input_param.getTypedData(&data, Header(i * ms));
                                boost::this_fiber::sleep_for(std::chrono::nanoseconds(5));
                            }
                        }
                        else
                        {
                            int new_value;
                            result = input_param.getTypedData(&new_value, Header(i * ms));
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
                        // boost::this_thread::sleep_for(boost::chrono::nanoseconds(wait));
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
            // boost::thread thread(read_func);
            while (!started)
            {
                boost::this_fiber::sleep_for(std::chrono::nanoseconds(10));
            }
            for (int i = 0; i < 10000; ++i)
            {
                // TODO fix
                param.updateData(i, timestamp = i * ms);
                boost::this_fiber::sleep_for(std::chrono::nanoseconds(33));
            }
            thread.join();
            return read_values;
        }
    };
} // namespace

TEST_F(ParamBufferFixture, read_map)
{
    init(BufferFlags::MAP_BUFFER);
    testRead();
    buffer->setFrameBufferCapacity(10);
    fill(100);
    ASSERT_EQ(buffer->getSize(), 100);
    const auto read_values = testMultiThreaded(false);
    ASSERT_EQ(read_values, 10000);
}

TEST_F(ParamBufferFixture, read_stream)
{
    init(BufferFlags::STREAM_BUFFER);
    testRead();
    testPruning();
    const auto read_values = testMultiThreaded(false);
    ASSERT_EQ(read_values, 10000);
}

TEST_F(ParamBufferFixture, blocking_stream)
{
    init(BufferFlags::BLOCKING_STREAM_BUFFER);
    testRead();
    testPruning();
    const auto read_values = testMultiThreaded(false);
    ASSERT_EQ(read_values, 10000);
}

TEST_F(ParamBufferFixture, read_dropping_stream)
{
    init(BufferFlags::DROPPING_STREAM_BUFFER);
    testRead();
    testPruning();
    // TODO more reliable test, fails more than half the time depending on CPU load
    // const auto read_values = testMultiThreaded(true);
    // BOOST_REQUIRE_GT(read_values, 1000);
}

TEST_F(ParamBufferFixture, read_nearest_neighbor)
{
    init(BufferFlags::NEAREST_NEIGHBOR_BUFFER);
    testRead();

    int data = 10;
    ASSERT_EQ(input_param.getTypedData(&data, Header(4.0 * mo::ms)), true);
    ASSERT_EQ(data, 5);

    data = 10;
    ASSERT_EQ(input_param.getTypedData(&data, Header(6.0 * mo::ms)), true);
    ASSERT_EQ(data, 5);

    data = 10;
    ASSERT_EQ(input_param.getTypedData(&data, Header(1.6 * mo::ms)), true);
    ASSERT_EQ(data, 2);

    data = 10;
    ASSERT_EQ(input_param.getTypedData(&data, Header(1.2 * mo::ms)), true);
    ASSERT_EQ(data, 1);

    testPruning();
}
