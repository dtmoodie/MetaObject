#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

#include <MetaObject/params/ITInputParam.hpp>
#include <MetaObject/params/TInputParam.hpp>
#include <MetaObject/params/TParamPtr.hpp>
#include <MetaObject/params/buffers/BufferFactory.hpp>
#include <MetaObject/params/buffers/IBuffer.hpp>

#include <boost/thread.hpp>

using namespace mo;

struct Fixture
{
    int val = 0;
    ITInputParam<int> input_param;
    TParamPtr<int> param;

    std::unique_ptr<buffer::IBuffer> buffer;

    Fixture()
        : input_param("sub")
        , param("pub", &val)
    {
    }

    void init(BufferFlags buffer_type)
    {
        buffer.reset(buffer::BufferFactory::createBuffer(&param, buffer_type));
        BOOST_REQUIRE(buffer);
        BOOST_REQUIRE(input_param.setInput(buffer.get()));
    }

    void testRead()
    {
        int data;
        BOOST_REQUIRE(!input_param.getData(Header()));
        param.updateData(1, Header(Time(mo::ms * 1.0)));

        BOOST_REQUIRE_EQUAL(buffer->getSize(), 1);
        BOOST_REQUIRE(input_param.getData(Header()));
        BOOST_REQUIRE(input_param.getData(Header(Time(mo::ms * 1.0))));
        BOOST_REQUIRE(input_param.getTypedData(&data, Header(Time(mo::ms * 1.0))));
        BOOST_REQUIRE_EQUAL(data, 1);

        param.updateData(2, Header(Time(mo::ms * 2.0)));
        BOOST_REQUIRE_EQUAL(buffer->getSize(), 2);
        BOOST_REQUIRE(input_param.getTypedData(&data, Header(Time(mo::ms * 2.0))));
        BOOST_REQUIRE_EQUAL(data, 2);

        param.updateData(5, Header(Time(mo::ms * 5.0)));
        BOOST_REQUIRE_EQUAL(buffer->getSize(), 3);
        BOOST_REQUIRE(input_param.getTypedData(&data, Header(Time(mo::ms * 5.0))));
        BOOST_REQUIRE_EQUAL(data, 5);

        BOOST_REQUIRE(input_param.getData(Header()));
        BOOST_REQUIRE(input_param.getTypedData(&data, Header()));
        BOOST_REQUIRE_EQUAL(data, 5);

        data = 10;
        BOOST_REQUIRE(input_param.getTypedData(&data, Header(mo::ms * 5.0)));
        BOOST_REQUIRE_EQUAL(data, 5);
    }

    void fill(const int32_t num)
    {
        BOOST_REQUIRE(buffer && "Something wrong in test, buffer not initialized already");
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
        BOOST_REQUIRE_EQUAL(buffer->getSize(), 10);
    }

    void testMultiThreaded(const bool dropping)
    {
        int read_values = 0;
        buffer->clear();

        auto read_func = [this, dropping, &read_values]() {
            int data;
            for (int i = 0; i < 10000; ++i)
            {
                bool result = false;
                for (int j = 0; j < 10 && !result; ++j)
                {
                    result = input_param.getTypedData(&data, Header(i * ms));
                    boost::this_thread::sleep_for(boost::chrono::nanoseconds(15));
                }

                BOOST_REQUIRE(result || dropping);
                if (result)
                {
                    BOOST_REQUIRE_EQUAL(data, i);
                    ++read_values;
                }
                auto wait = rand() % 20 + 20;
                boost::this_thread::sleep_for(boost::chrono::nanoseconds(wait));
                if (boost::this_thread::interruption_requested())
                {
                    return;
                }
            }
        };
        boost::thread thread(read_func);

        for (int i = 0; i < 10000; ++i)
        {
            param.updateData(i, tag::_timestamp = i * ms);
            boost::this_thread::sleep_for(boost::chrono::nanoseconds(33));
        }
        thread.join();
        BOOST_REQUIRE(dropping || read_values == 10000);
    }
};

BOOST_FIXTURE_TEST_CASE(TestReadMap, Fixture)
{
    init(MAP_BUFFER);
    testRead();
    buffer->setFrameBufferCapacity(10);
    fill(100);
    BOOST_REQUIRE_EQUAL(buffer->getSize(), 100);
    testMultiThreaded(false);
}

BOOST_FIXTURE_TEST_CASE(TestReadStream, Fixture)
{
    init(STREAM_BUFFER);
    testRead();
    testPruning();
    testMultiThreaded(false);
}

BOOST_FIXTURE_TEST_CASE(TestReadBlockingStream, Fixture)
{
    init(BLOCKING_STREAM_BUFFER);
    testRead();
    testPruning();
    testMultiThreaded(false);
}

BOOST_FIXTURE_TEST_CASE(TestReadDroppingStream, Fixture)
{
    init(DROPPING_STREAM_BUFFER);
    testRead();
    testPruning();
    testMultiThreaded(true);
}

BOOST_FIXTURE_TEST_CASE(TestReadNearestNeighbor, Fixture)
{
    init(NEAREST_NEIGHBOR_BUFFER);
    testRead();

    int data = 10;
    BOOST_REQUIRE(input_param.getTypedData(&data, Header(4.0 * mo::ms)));
    BOOST_REQUIRE_EQUAL(data, 5);

    data = 10;
    BOOST_REQUIRE(input_param.getTypedData(&data, Header(6.0 * mo::ms)));
    BOOST_REQUIRE_EQUAL(data, 5);

    data = 10;
    BOOST_REQUIRE(input_param.getTypedData(&data, Header(1.6 * mo::ms)));
    BOOST_REQUIRE_EQUAL(data, 2);

    data = 10;
    BOOST_REQUIRE(input_param.getTypedData(&data, Header(1.2 * mo::ms)));
    BOOST_REQUIRE_EQUAL(data, 1);

    testPruning();
}
