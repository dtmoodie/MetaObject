#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Signals/TypedSignal.hpp"
#include "MetaObject/Detail/Counter.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"
#include "MetaObject/Signals/detail/SignalMacros.hpp"
#include "MetaObject/Signals/detail/SlotMacros.hpp"
#include "MetaObject/Parameters//ParameterMacros.hpp"
#include "MetaObject/Parameters/TypedParameterPtr.hpp"
#include "MetaObject/Parameters/TypedInputParameter.hpp"
#include "RuntimeObjectSystem.h"
#include "IObjectFactorySystem.h"


#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "signals"
#include <boost/test/unit_test.hpp>
#include <iostream>

using namespace mo;

BOOST_AUTO_TEST_CASE(signals)
{
	TypedSignal<int(int)> signal;
	{
		TypedSlot<int(int)> slot([](int val)
		{
			return val * 2;
		});
		signal.Connect(&slot);

		BOOST_CHECK_EQUAL(signal(4), 8);
	}
	BOOST_CHECK_THROW(signal(4), std::string);
	
	
}