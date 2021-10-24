#include "MetaObject/core/SystemTable.hpp"
#include "MetaObject/logging/CompileLogger.hpp"
#include "MetaObject/object/MetaObject.hpp"
#include "MetaObject/object/detail/MetaObjectMacros.hpp"
#include "MetaObject/params/ParamMacros.hpp"
#include "MetaObject/params/TSubscriberPtr.hpp"
#include "MetaObject/params/buffers/BufferFactory.hpp"
#include "MetaObject/params/detail/MetaParamImpl.hpp"
#include "MetaObject/signals/TSignal.hpp"
#include "MetaObject/signals/detail/SignalMacros.hpp"
#include "MetaObject/signals/detail/SlotMacros.hpp"
#include "RuntimeObjectSystem/IObjectFactorySystem.h"
#include "RuntimeObjectSystem/RuntimeObjectSystem.h"

#include <gtest/gtest.h>

#include "TestObjects.hpp"

#include <boost/thread.hpp>

#include <iostream>
#include <thread>
using namespace mo;

using namespace test;

TEST(object, input_parameter_manual)
{
    IAsyncStreamPtr_t stream = IAsyncStream::create();
    mo::MetaObjectFactory::instance();
    auto input = InputParameterizedObject::create();
    auto output = OutputParameterizedObject::create();
    ASSERT_EQ(input->test_input_param.setInput(&output->test_output), true);
    output->increment();
    ASSERT_NE(input->test_input, nullptr);
    // ASSERT_EQ(*input->test_input, 10);
    ASSERT_EQ(*input->test_input, output->output_val);
}

TEST(object, input_parameter_programatic)
{
    IAsyncStreamPtr_t stream = IAsyncStream::create();
    auto input = InputParameterizedObject::create();
    auto input_ = input->getInput("test_input");

    auto output = OutputParameterizedObject::create();
    auto output_ = output->getOutput("test_output");

    ASSERT_NE(input_, nullptr);
    ASSERT_NE(output_, nullptr);
    auto input_param = dynamic_cast<ISubscriber*>(input_);
    ASSERT_NE(input_param, nullptr);
    ASSERT_EQ(input_param->setInput(output_), true);
    output->increment();
    ASSERT_NE(input->test_input, nullptr);
    ASSERT_EQ(*input->test_input, output->output_val);
}

/*BOOST_AUTO_TEST_CASE(buffered_input)
{
    rcc::shared_ptr<input_parametered_object> input = input_parametered_object::create();
    auto input_ = input->getParam("test_input");

    auto output = output_parametered_object::create();
    auto output_ = output->getParam("test_output");

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
    auto input_ = input->getParam("test_input");

    auto output = output_parametered_object::create();
    auto output_ = output->getParam("test_output");

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
    IAsyncStreamPtr_t stream = IAsyncStream::create();
    auto input = InputParameterizedObject::create();
    auto input_param = input->getInput("test_input");

    auto output = OutputParameterizedObject::create();
    auto output_ = output->getOutput("test_output");

    ASSERT_NE(input_param, nullptr);
    ASSERT_NE(output_, nullptr);

    auto buffer = buffer::IBuffer::create(mo::BufferFlags::STREAM_BUFFER);
    ASSERT_NE(buffer, nullptr);
    buffer->setInput(output_);
    ASSERT_EQ(input_param->setInput(buffer), true);
    output->test_output.publish(0, mo::Header(0));

    std::thread background_thread([&input]() {
        for (int i = 0; i < 1000; ++i)
        {
            mo::Header header(i);
            auto container = input->test_input_param.getData(&header);

            while (!container)
            {
                container = input->test_input_param.getData(&header);
            }
            auto typed = std::dynamic_pointer_cast<const TDataContainer<int>>(container);
            ASSERT_TRUE(typed);
            ASSERT_EQ(typed->data, i * 10);
            boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
        }
    });

    for (int i = 1; i < 1000; ++i)
    {
        output->test_output.publish(i * 10, mo::Header(i));
    }
    background_thread.join();
}
