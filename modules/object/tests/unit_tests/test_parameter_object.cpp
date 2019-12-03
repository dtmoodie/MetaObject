#include "MetaObject/core/SystemTable.hpp"
#include "MetaObject/logging/CompileLogger.hpp"
#include "MetaObject/object/MetaObject.hpp"
#include "MetaObject/object/detail/MetaObjectMacros.hpp"
#include "MetaObject/params/ParamMacros.hpp"
#include "MetaObject/params/TInputParam.hpp"
#include "MetaObject/params/TParamPtr.hpp"
#include "MetaObject/params/buffers/BufferFactory.hpp"
#include "MetaObject/params/detail/MetaParamImpl.hpp"
#include "MetaObject/signals/TSignal.hpp"
#include "MetaObject/signals/detail/SignalMacros.hpp"
#include "MetaObject/signals/detail/SlotMacros.hpp"
#include "RuntimeObjectSystem/IObjectFactorySystem.h"
#include "RuntimeObjectSystem/RuntimeObjectSystem.h"

#include <gtest/gtest.h>

#include <RuntimeObjectSystem/RuntimeLinkLibrary.h>

#include <boost/thread.hpp>

#include <iostream>
#include <thread>
using namespace mo;

struct output_parametered_object : public MetaObject
{
    MO_BEGIN(output_parametered_object)
    OUTPUT(int, test_output, 0)
    OUTPUT(double, test_double, 0.0)
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
    MO_BEGIN(input_parametered_object)
    INPUT(int, test_input)
    MO_END;
};

MO_REGISTER_OBJECT(input_parametered_object)
MO_REGISTER_OBJECT(output_parametered_object)

TEST(object, input_parameter_manual)
{
    mo::MetaObjectFactory::instance();
    auto input = input_parametered_object::create();
    auto output = output_parametered_object::create();
    ASSERT_EQ(input->test_input_param.setInput(&output->test_output), true);
    ASSERT_NE(input->test_input, nullptr);
    ASSERT_EQ(*(input->test_input), 0);
    output->test_output.updateData(10);
    ASSERT_NE(input->test_input, nullptr);
    ASSERT_EQ(*input->test_input, 10);

    ASSERT_EQ(*input->test_input, output->test_output.access()());
}

TEST(object, input_parameter_programatic)
{
    auto input = input_parametered_object::create();
    auto input_ = input->getParamOptional("test_input");

    auto output = output_parametered_object::create();
    auto output_ = output->getParamOptional("test_output");

    ASSERT_NE(input_, nullptr);
    ASSERT_NE(output_, nullptr);
    auto input_param = dynamic_cast<InputParam*>(input_);
    ASSERT_NE(input_param, nullptr);
    ASSERT_EQ(input_param->setInput(output_), true);
    ASSERT_NE(input->test_input, nullptr);
    ASSERT_EQ(*(input->test_input), 0);
    output->test_output.updateData(10);
    ASSERT_NE(input->test_input, nullptr);
    ASSERT_EQ(*input->test_input, output->test_output.access()());
}

/*BOOST_AUTO_TEST_CASE(buffered_input)
{
    rcc::shared_ptr<input_parametered_object> input = input_parametered_object::create();
    auto input_ = input->getParamOptional("test_input");

    auto output = output_parametered_object::create();
    auto output_ = output->getParamOptional("test_output");

    ASSERT_TRUE(input_);
    ASSERT_TRUE(output_);
    auto input_param = dynamic_cast<InputParam*>(input_);

    auto cbuffer = buffer::BufferFactory::createBuffer(output_, mo::BufferFlags::CIRCULAR_BUFFER);
    ASSERT_TRUE(cbuffer);
    ASSERT_TRUE(input_param->setInput(cbuffer));
    output->test_output_param.updateData(0, 0);
    for (int i = 1; i < 100000; ++i)
    {
        output->test_output_param.updateData(i * 10, i);
        auto data = input->test_input_param.template getTypedData<int>(i - 1);
        ASSERT_TRUE(data);
        ASSERT_EQ(data->data, (i - 1) * 10);
    }
}

BOOST_AUTO_TEST_CASE(threaded_buffered_input)
{
    auto input = input_parametered_object::create();
    auto input_ = input->getParamOptional("test_input");

    auto output = output_parametered_object::create();
    auto output_ = output->getParamOptional("test_output");

    ASSERT_TRUE(input_);
    ASSERT_TRUE(output_);
    auto input_param = dynamic_cast<InputParam*>(input_);

    auto cbuffer = buffer::BufferFactory::createBuffer(output_, mo::BufferFlags::CIRCULAR_BUFFER);
    ASSERT_TRUE(cbuffer);
    ASSERT_TRUE(input_param->setInput(cbuffer));
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
                ASSERT_EQ(container->data, ts * 10);
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

TEST(object, threaded_stream_buffer)
{
    auto input = input_parametered_object::create();
    auto input_ = input->getParamOptional("test_input");

    auto output = output_parametered_object::create();
    auto output_ = output->getParamOptional("test_output");

    ASSERT_NE(input_, nullptr);
    ASSERT_NE(output_, nullptr);
    auto input_param = dynamic_cast<InputParam*>(input_);

    auto buffer = buffer::BufferFactory::instance()->createBuffer(output_, mo::BufferFlags::STREAM_BUFFER);
    ASSERT_NE(buffer, nullptr);
    ASSERT_EQ(input_param->setInput(static_cast<mo::IParam*>(buffer)), true);
    output->test_output.updateData(0, 0);

    std::thread background_thread([&input]() {
        for (int i = 0; i < 1000; ++i)
        {
            auto container = input->test_input_param.getTypedData<int>(i);

            while (!container)
            {
                container = input->test_input_param.getTypedData<int>(i);
            }
            ASSERT_EQ(container->data, i * 10);
            boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
        }
    });

    for (int i = 1; i < 1000; ++i)
    {
        output->test_output.updateData(i * 10, i);
    }
    background_thread.join();
}
