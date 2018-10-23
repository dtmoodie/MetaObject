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
        BOOST_REQUIRE(!input_param.getInputData(Header()));
        param.updateData(1, Header(Time(mo::ms * 1.0)));
        BOOST_REQUIRE_EQUAL(buffer->getSize(), 1);
        BOOST_REQUIRE(input_param.getInputData(Header()));
        BOOST_REQUIRE(input_param.getInputData(Header(Time(mo::ms * 1.0))));
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
}
