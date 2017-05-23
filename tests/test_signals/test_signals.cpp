#define BOOST_TEST_MAIN

#include "MetaObject/object/IMetaObject.hpp"
#include "MetaObject/signals/TSignal.hpp"
#include "MetaObject/signals/RelayManager.hpp"
#include "MetaObject/detail/Counter.hpp"
#include "MetaObject/detail/MetaObjectMacros.hpp"
#include "MetaObject/signals/detail/SignalMacros.hpp"
#include "MetaObject/signals/detail/SlotMacros.hpp"
#include "MetaObject/params//ParamMacros.hpp"
#include "MetaObject/params/TParamPtr.hpp"
#include "MetaObject/params/TInputParam.hpp"

#include "RuntimeObjectSystem/RuntimeObjectSystem.h"
#include "RuntimeObjectSystem/IObjectFactorySystem.h"


#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE __FILE__
#include <boost/test/included/unit_test.hpp>
#endif
#include <boost/thread.hpp>
#include <iostream>

using namespace mo;

BOOST_AUTO_TEST_CASE(signals)
{
	TSignal<int(int)> signal;
	{
		TSlot<int(int)> slot([](int val)
		{
			return val * 2;
		});
		signal.connect(&slot);

		BOOST_CHECK_EQUAL(signal(4), 8);
	}
	BOOST_CHECK_THROW(signal(4), std::string);	
}

BOOST_AUTO_TEST_CASE(threaded_signal)
{
    mo::Context ctx;
    mo::Context thread_ctx;

    TSlot<void(int)> slot = TSlot<void(int)>(std::bind(
        [&thread_ctx](int value)->void
        {
            BOOST_REQUIRE_EQUAL(thread_ctx.thread_id, mo::getThisThread());
            BOOST_REQUIRE_EQUAL(5, value);
        }, std::placeholders::_1));

    slot.setContext(&thread_ctx);

    TSignal<void(int)> signal;
    auto Connection = slot.connect(&signal);

    boost::thread thread = boost::thread([&thread_ctx]()->void
    {
        thread_ctx.thread_id = mo::getThisThread();
        while(!boost::this_thread::interruption_requested())
        {
            ThreadSpecificQueue::Run(thread_ctx.thread_id);
        }
    });

    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
    signal(&ctx, 5);
    thread.interrupt();
    thread.join();
}

BOOST_AUTO_TEST_CASE(relay_manager)
{
    mo::Context ctx;
    mo::RelayManager manager;

    
}
