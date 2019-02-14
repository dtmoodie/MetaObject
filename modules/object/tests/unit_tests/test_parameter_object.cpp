#include "MetaObject/core/SystemTable.hpp"
#include "MetaObject/logging/CompileLogger.hpp"
#include "MetaObject/object/MetaObject.hpp"
#include "MetaObject/object/detail/MetaObjectMacros.hpp"
#include "MetaObject/params/ParamMacros.hpp"
#include "MetaObject/params/TInputParam.hpp"
#include "MetaObject/params/TParamPtr.hpp"
#include "MetaObject/params/buffers/BufferFactory.hpp"
#include "MetaObject/params/buffers/CircularBuffer.hpp"
#include "MetaObject/params/buffers/Map.hpp"
#include "MetaObject/params/buffers/StreamBuffer.hpp"
#include "MetaObject/params/detail/MetaParamImpl.hpp"
#include "MetaObject/signals/TSignal.hpp"
#include "MetaObject/signals/detail/SignalMacros.hpp"
#include "MetaObject/signals/detail/SlotMacros.hpp"
#include "RuntimeObjectSystem/IObjectFactorySystem.h"
#include "RuntimeObjectSystem/RuntimeObjectSystem.h"

#include <boost/test/auto_unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <boost/thread.hpp>

#include <iostream>
#include <thread>
using namespace mo;

struct output_parametered_object : public MetaObject
{
    MO_BEGIN
    OUTPUT(int, test_output, 0);
    OUTPUT(double, test_double, 0.0);
    MO_END;
    void increment()
    {
        // wut, get access token, get ref to data, increment
        test_output.access()()++;
        // old way
        // test_output++;
    }
};

struct input_parametered_object : public MetaObject
{
    MO_BEGIN
    INPUT(int, test_input)
    MO_END;
};

MO_REGISTER_OBJECT(input_parametered_object)
MO_REGISTER_OBJECT(output_parametered_object)

CompileLogger* logger = nullptr;
BuildCallback* cb = nullptr;
BOOST_AUTO_TEST_CASE(input_parameter_manual)
{
    mo::MetaObjectFactory::instance();
    auto input = input_parametered_object::create();
    auto output = output_parametered_object::create();
    BOOST_REQUIRE(input->test_input_param.setInput(&output->test_output));
    BOOST_REQUIRE(input->test_input == nullptr);
    output->test_output.updateData(10);
    BOOST_REQUIRE(input->test_input != nullptr);
    BOOST_REQUIRE(*input->test_input == 10);

    BOOST_REQUIRE_EQUAL(*input->test_input, output->test_output.access()());
}

BOOST_AUTO_TEST_CASE(input_parameter_programatic)
{
    auto input = input_parametered_object::create();
    auto input_ = input->getParamOptional("test_input");

    auto output = output_parametered_object::create();
    auto output_ = output->getParamOptional("test_output");

    BOOST_REQUIRE(input_);
    BOOST_REQUIRE(output_);
    auto input_param = dynamic_cast<InputParam*>(input_);
    BOOST_REQUIRE(input_param);
    BOOST_REQUIRE(input_param->setInput(output_));
    BOOST_REQUIRE(input->test_input == nullptr);
    output->test_output.updateData(10);
    BOOST_REQUIRE(input->test_input != nullptr);
    BOOST_REQUIRE_EQUAL(*input->test_input, output->test_output.access()());
}

/*BOOST_AUTO_TEST_CASE(buffered_input)
{
    rcc::shared_ptr<input_parametered_object> input = input_parametered_object::create();
    auto input_ = input->getParamOptional("test_input");

    auto output = output_parametered_object::create();
    auto output_ = output->getParamOptional("test_output");

    BOOST_REQUIRE(input_);
    BOOST_REQUIRE(output_);
    auto input_param = dynamic_cast<InputParam*>(input_);

    auto cbuffer = buffer::BufferFactory::createBuffer(output_, mo::BufferFlags::CIRCULAR_BUFFER);
    BOOST_REQUIRE(cbuffer);
    BOOST_REQUIRE(input_param->setInput(cbuffer));
    output->test_output_param.updateData(0, 0);
    for (int i = 1; i < 100000; ++i)
    {
        output->test_output_param.updateData(i * 10, i);
        auto data = input->test_input_param.template getTypedData<int>(i - 1);
        BOOST_REQUIRE(data);
        BOOST_REQUIRE_EQUAL(data->data, (i - 1) * 10);
    }
}

BOOST_AUTO_TEST_CASE(threaded_buffered_input)
{
    auto input = input_parametered_object::create();
    auto input_ = input->getParamOptional("test_input");

    auto output = output_parametered_object::create();
    auto output_ = output->getParamOptional("test_output");

    BOOST_REQUIRE(input_);
    BOOST_REQUIRE(output_);
    auto input_param = dynamic_cast<InputParam*>(input_);

    auto cbuffer = buffer::BufferFactory::createBuffer(output_, mo::BufferFlags::CIRCULAR_BUFFER);
    BOOST_REQUIRE(cbuffer);
    BOOST_REQUIRE(input_param->setInput(cbuffer));
    output->test_output_param.updateData(0, 0);

    bool quit = false;
    bool stopped = false;
    std::thread background_thread([&stopped, &quit, &input]() {
        int ts = 0;
        int data;
        while (!quit)
        {
            auto container = input->test_input_param.getTypedData<int>(ts);
            if (container)
            {
                BOOST_REQUIRE_EQUAL(container->data, ts * 10);
                ++ts;
            }
        }
        stopped = true;
    });

    for (int i = 1; i < 100000; ++i)
    {
        output->test_output_param.updateData(i * 10, i);
    }
    quit = true;
    while (!stopped)
    {
        boost::this_thread::sleep_for(boost::chrono::milliseconds(5));
    }
    background_thread.join();
}*/

BOOST_AUTO_TEST_CASE(threaded_stream_buffer)
{
    auto input = input_parametered_object::create();
    auto input_ = input->getParamOptional("test_input");

    auto output = output_parametered_object::create();
    auto output_ = output->getParamOptional("test_output");

    BOOST_REQUIRE(input_);
    BOOST_REQUIRE(output_);
    auto input_param = dynamic_cast<InputParam*>(input_);

    auto buffer = buffer::BufferFactory::createBuffer(output_, mo::STREAM_BUFFER);
    BOOST_REQUIRE(buffer);
    BOOST_REQUIRE(input_param->setInput(buffer));
    output->test_output.updateData(0, 0);

    std::thread background_thread([&input]() {
        for (int i = 0; i < 1000; ++i)
        {
            auto container = input->test_input_param.getTypedData<int>(i);

            while (!container)
            {
                container = input->test_input_param.getTypedData<int>(i);
            }
            BOOST_REQUIRE_EQUAL(container->data, i * 10);
            boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
        }
    });

    for (int i = 1; i < 1000; ++i)
    {
        output->test_output.updateData(i * 10, i);
    }
    background_thread.join();
}

BOOST_AUTO_TEST_CASE(cleanup)
{
}
