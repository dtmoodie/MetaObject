#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Signals/TypedSignal.hpp"
#include "MetaObject/Detail/Counter.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"
#include "MetaObject/Signals/detail/SignalMacros.hpp"
#include "MetaObject/Signals/detail/SlotMacros.hpp"
#include "MetaObject/Signals/detail/CallbackMacros.hpp"
#include "MetaObject/Parameters//ParameterMacros.hpp"
#include "MetaObject/Parameters/TypedParameterPtr.hpp"
#include "MetaObject/Parameters/TypedInputParameter.hpp"
#include "RuntimeObjectSystem.h"
#include "IObjectFactorySystem.h"


#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "parameter"
#include <boost/test/unit_test.hpp>
#include <iostream>

using namespace mo;

BOOST_AUTO_TEST_CASE(wrapped_parameter)
{
	int value = 10;
	TypedParameterPtr<int> param("Test wrapped param", &value);

	BOOST_CHECK_EQUAL(param.GetData(), 10);
	param.UpdateData(5);
	BOOST_CHECK_EQUAL(param.GetData(), 5);
	value = 11;
	BOOST_CHECK_EQUAL(param.GetData(), 11);
	bool update_handler_called = false;
	TypedSlot<void(Context*, IParameter*)> slot([&param, &update_handler_called](Context* ctx, IParameter* param_in)
	{
		update_handler_called = param_in == &param;
	});
	param.RegisterUpdateNotifier(&slot);
	param.UpdateData(5);
	BOOST_REQUIRE_EQUAL(update_handler_called, true);

	
}

BOOST_AUTO_TEST_CASE(input_parameter)
{
	int value = 10;
	TypedParameterPtr<int> param("Test wrapped param", &value);
	ITypedInputParameter<int> input_param;
	BOOST_REQUIRE(input_param.SetInput(&param));
	BOOST_REQUIRE_EQUAL(input_param.GetData(), value);
	
	bool update_handler_called = false;
	TypedSlot<void(Context*, IParameter*)> slot(
		[&update_handler_called](Context*, IParameter*)
	{
		update_handler_called = true;
	});

	BOOST_REQUIRE(input_param.RegisterUpdateNotifier(&slot));
	param.UpdateData(5);
	BOOST_REQUIRE_EQUAL(update_handler_called, true);
}



