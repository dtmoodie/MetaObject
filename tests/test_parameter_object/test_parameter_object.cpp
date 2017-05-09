#define BOOST_TEST_MAIN
#include "MetaObject/Params/MetaParam.hpp"
#include "MetaObject/Params/Buffers/CircularBuffer.hpp"
#include "MetaObject/Params/Buffers/StreamBuffer.hpp"
#include "MetaObject/Params/Buffers/Map.hpp"
#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Signals/TSignal.hpp"
#include "MetaObject/Detail/Counter.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"
#include "MetaObject/Signals/Detail/SignalMacros.hpp"
#include "MetaObject/Signals/Detail/SlotMacros.hpp"
#include "MetaObject/Params/ParamMacros.hpp"
#include "MetaObject/Params/TParamPtr.hpp"
#include "MetaObject/Params/TInputParam.hpp"
#include "MetaObject/Logging/CompileLogger.hpp"
#include "MetaObject/Params/Buffers/BufferFactory.hpp"
#include "MetaObject/Detail/Allocator.hpp"
#include "MetaObject/Params/Detail/MetaParamImpl.hpp"
#include "RuntimeObjectSystem/RuntimeObjectSystem.h"
#include "RuntimeObjectSystem/IObjectFactorySystem.h"

#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE "Param_object"
#include <boost/test/included/unit_test.hpp>
#endif

#include <boost/thread.hpp>
#include <iostream>
#include <fstream>
using namespace mo;

INSTANTIATE_META_Param(int);



struct output_Paramed_object: public IMetaObject
{
    MO_BEGIN(output_Paramed_object)
        OUTPUT(int, test_output, 0);
        OUTPUT(double, test_double, 0.0);
    MO_END;
    void increment()
    {
        test_output++;
    }
};

struct input_Paramed_object: public IMetaObject
{
    MO_BEGIN(input_Paramed_object)
        INPUT(int, test_input, nullptr)
    MO_END;
};

MO_REGISTER_OBJECT(input_Paramed_object)
MO_REGISTER_OBJECT(output_Paramed_object)


BuildCallback* cb = nullptr;
BOOST_AUTO_TEST_CASE(input_Param_manual)
{
    MetaObjectFactory::Instance()->GetObjectSystem()->SetupObjectConstructors(PerModuleInterface::GetInstance());
    cb = new BuildCallback();
    auto input = input_Paramed_object::Create();
    auto output = output_Paramed_object::Create();
    input->test_input_param.setInput(&output->test_output_param);
    BOOST_REQUIRE(input->test_input);
    BOOST_REQUIRE_EQUAL(input->test_input, &output->test_output);
    BOOST_REQUIRE_EQUAL(*input->test_input, output->test_output);    
}



BOOST_AUTO_TEST_CASE(input_Param_programatic)
{
    auto input = input_Paramed_object::Create();
    auto input_ = input->getParamOptional("test_input");

    auto output = output_Paramed_object::Create();
    auto output_ = output->getParamOptional("test_output");
    
    BOOST_REQUIRE(input_);
    BOOST_REQUIRE(output_);
    auto input_param = dynamic_cast<InputParam*>(input_);
    BOOST_REQUIRE(input_param);
    BOOST_REQUIRE(input_param->setInput(output_));
    BOOST_REQUIRE(input->test_input);
    BOOST_REQUIRE_EQUAL(input->test_input, &output->test_output);
    BOOST_REQUIRE_EQUAL(*input->test_input, output->test_output);
}

BOOST_AUTO_TEST_CASE(buffered_input)
{
    auto input = input_Paramed_object::Create();
    auto input_ = input->getParamOptional("test_input");

    auto output = output_Paramed_object::Create();
    auto output_ = output->getParamOptional("test_output");

    BOOST_REQUIRE(input_);
    BOOST_REQUIRE(output_);
    auto input_param = dynamic_cast<InputParam*>(input_);

    auto cbuffer = Buffer::BufferFactory::CreateProxy(output_, CircularBuffer_e);
    BOOST_REQUIRE(cbuffer);
    BOOST_REQUIRE(input_param->setInput(cbuffer));
    output->test_output_param.updateData(0, 0);
    for(int i = 1; i < 100000; ++i)
    {
        output->test_output_param.updateData(i*10, mo::Time_t(i * mo::ms));
        int data;
        BOOST_REQUIRE(input->test_input_param.getData(data, mo::Time_t((i-1) * mo::ms)));
        BOOST_REQUIRE_EQUAL(data, (i-1)*10);
    }
}

BOOST_AUTO_TEST_CASE(threaded_buffered_input)
{
    auto input = input_Paramed_object::Create();
    auto input_ = input->getParamOptional("test_input");

    auto output = output_Paramed_object::Create();
    auto output_ = output->getParamOptional("test_output");

    BOOST_REQUIRE(input_);
    BOOST_REQUIRE(output_);
    auto input_param = dynamic_cast<InputParam*>(input_);

    auto cbuffer = Buffer::BufferFactory::CreateProxy(output_, CircularBuffer_e);
    BOOST_REQUIRE(cbuffer);
    BOOST_REQUIRE(input_param->setInput(cbuffer));
    output->test_output_param.updateData(0, 0);
    bool quit = false;
    std::thread background_thread(
    [&quit,&input]()
    {
        mo::Time_t ts = mo::Time_t(0 * mo::ms);
        int data;
        while(!quit)
        {
            if(input->test_input_param.getData(data, ts))
            {
                //BOOST_REQUIRE_EQUAL(mo::time_t(data * mo::ms), mo::time_t(ts * (10 * mo::ms)));
                ts += mo::Time_t(1 * mo::ms);
            }
        }
    });

    for(int i = 1; i < 100000; ++i)
    {
        output->test_output_param.updateData(i*10, mo::Time_t(i*mo::ms));
    }
    quit = true;
    background_thread.join();
}

BOOST_AUTO_TEST_CASE(threaded_stream_buffer)
{
    mo::Context ctx;
    auto input = input_Paramed_object::Create();
    auto input_ = input->getParamOptional("test_input");

    auto output = output_Paramed_object::Create();
    output->setContext(&ctx);
    auto output_ = output->getParamOptional("test_output");

    BOOST_REQUIRE(input_);
    BOOST_REQUIRE(output_);
    auto input_param = dynamic_cast<InputParam*>(input_);

    auto buffer = Buffer::BufferFactory::CreateProxy(output_, StreamBuffer_e);
    BOOST_REQUIRE(buffer);
    BOOST_REQUIRE(input_param->setInput(buffer));
    output->test_output_param.updateData(0, 0);
    volatile bool started = false;
    volatile int count = 0;
    boost::thread background_thread(
        [&input, &started, &count]()
    {
        mo::Context _ctx;
        input->setContext(&_ctx);
        started = true;
        int data;
        for(int i = 0; i < 1000; ++i)
        {
            bool good = input->test_input_param.getData(data, mo::Time_t(i*mo::ms));
            while(!good)
            {
                good = input->test_input_param.getData(data, mo::Time_t(i*mo::ms));
            }
            ++count;
            BOOST_REQUIRE_EQUAL(data, i * 10);
            boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
        }
        mo::Allocator::CleanupThreadSpecificAllocator();
    });
    while(!started)
    {
        boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
    }
    for(int i = 1; i < 1000; ++i)
    {
        output->test_output_param.updateData(i*10, i);
        boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
    }
    std::cout << "Waiting for background thread to close\n";
    background_thread.interrupt();
    background_thread.timed_join(boost::posix_time::time_duration(0,2,0));
    //background_thread.();
    mo::Allocator::CleanupThreadSpecificAllocator();
}


BOOST_AUTO_TEST_CASE(cleanup)
{
    delete cb;
}



