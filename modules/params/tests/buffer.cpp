#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

#include <MetaObject/params/ITInputParam.hpp>
#include <MetaObject/params/TInputParam.hpp>
#include <MetaObject/params/TParamPtr.hpp>
#include <MetaObject/params/buffers/BufferFactory.hpp>
#include <MetaObject/params/buffers/IBuffer.hpp>

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
};

BOOST_FIXTURE_TEST_CASE(TestReadCircular, Fixture)
{
    init(CIRCULAR_BUFFER);
    testRead();
}

BOOST_FIXTURE_TEST_CASE(TestReadMap, Fixture)
{
    init(MAP_BUFFER);
    testRead();
}

BOOST_FIXTURE_TEST_CASE(TestReadStream, Fixture)
{
    init(STREAM_BUFFER);
    testRead();
}

BOOST_FIXTURE_TEST_CASE(TestReadBlockingStream, Fixture)
{
    init(BLOCKING_STREAM_BUFFER);
    testRead();
}

BOOST_FIXTURE_TEST_CASE(TestReadNearestNeighbor, Fixture)
{
    init(NEAREST_NEIGHBOR_BUFFER);
    testRead();

    int data = 10;
    BOOST_REQUIRE(input_param.getTypedData(&data, Header(mo::ms * 4.0)));
    BOOST_REQUIRE_EQUAL(data, 5);

    data = 10;
    BOOST_REQUIRE(input_param.getTypedData(&data, Header(mo::ms * 6.0)));
    BOOST_REQUIRE_EQUAL(data, 5);

    data = 10;
    BOOST_REQUIRE(input_param.getTypedData(&data, Header(mo::ms * 1.6)));
    BOOST_REQUIRE_EQUAL(data, 2);
}
